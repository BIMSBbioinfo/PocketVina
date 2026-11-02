#include "cache.h"
#include "wrapcl.h"
#include "monte_carlo.h"
#include "coords.h"
#include "mutate.h"
#include "quasi_newton.h"
#include "parallel_mc.h"
#include "szv_grid.h"
#include <thread>
#include <boost/progress.hpp>

#include "commonMacros.h"
#include "wrapcl.h"
#include "random.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <system_error>
#include <experimental/filesystem>

#include <stdio.h>

using namespace std;

/**************************************************************************/
/*************************    OpenCL kernel cache    **********************/
/**************************************************************************/
// <opencl_binary_path>/Kernel{1,2}_Opt.bin are loaded when both exist and the driver accepts them; otherwise both
// kernels are built from the sources under $POCKETVINA_KERNEL_SRC (default ".": OpenCL/src/kernels/*.cl and
// OpenCL/inc) and saved there, unless that is the working directory or inside the kernel source directory.
// Building the engine with -DBUILD_KERNEL_FROM_SOURCE always rebuilds.

static const char* const kernel_binary_names[2] = { "Kernel1_Opt.bin", "Kernel2_Opt.bin" };

// The order of files is important!
static const std::vector<std::string> kernel1_sources = { "code_head.cl", "kernel1.cl" };
static const std::vector<std::string> kernel2_sources = { "code_head.cl", "mutate_conf.cl", "matrix.cl", "visited.cl",
														  "quasi_newton.cl", "kernel2.cl" };

static bool is_regular_file(const std::string& path) {
	std::error_code ec;
	return std::experimental::filesystem::is_regular_file(path, ec);
}

// True if path is root or inside it; a path that does not exist yet is judged by its nearest existing parent.
static bool is_inside(const std::string& path, const std::string& root) {
	namespace fs = std::experimental::filesystem;
	std::error_code ec;
	fs::path p = fs::absolute(path);
	while (!fs::exists(p, ec) && p.has_relative_path()) p = p.parent_path();
	const fs::path a = fs::canonical(p, ec);
	if (ec) return false;
	const fs::path b = fs::canonical(root, ec);
	if (ec) return false;
	fs::path::const_iterator ia = a.begin();
	for (fs::path::const_iterator ib = b.begin(); ib != b.end(); ++ib, ++ia)
		if (ia == a.end() || *ia != *ib) return false;
	return true;
}

// NULL if the file cannot be read or the driver rejects it (e.g. built for another device or driver version).
static cl_program load_program_binary(cl_context context, cl_device_id* devices, const std::string& path) {
	std::ifstream f(path.c_str(), std::ios::in | std::ios::binary);
	if (!f) return NULL;
	std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (data.empty()) return NULL;
	const unsigned char* binary = (const unsigned char*)data.data();
	size_t size = data.size();
	cl_int binary_status = CL_SUCCESS, err;
	cl_program program = clCreateProgramWithBinary(context, 1, devices, &size, &binary, &binary_status, &err);
	if (err != CL_SUCCESS || binary_status != CL_SUCCESS) {
		if (program != NULL) clReleaseProgram(program);
		return NULL;
	}
	if (clBuildProgram(program, 1, devices, NULL, NULL, NULL) != CL_SUCCESS) {
		clReleaseProgram(program);
		return NULL;
	}
	return program;
}

static cl_program build_program_from_source(cl_context context, cl_device_id* devices, const std::string& src_root,
											const std::vector<std::string>& files) {
	std::string source;
	for (size_t i = 0; i < files.size(); i++) {
		const std::string path = src_root + "/OpenCL/src/kernels/" + files[i];
		std::ifstream f(path.c_str(), std::ios::in | std::ios::binary);
		std::stringstream text;
		if (f) text << f.rdbuf();
		if (!f || text.str().empty())
			throw std::runtime_error("cannot read the OpenCL kernel source \"" + path + "\"; set POCKETVINA_KERNEL_SRC "
									 "to the directory that holds OpenCL/src/kernels and OpenCL/inc");
		if (i > 0) source += '\n';
		source += text.str();
	}
	const char* source_ptr = source.c_str();
	size_t source_size = source.size();
	cl_int err;
	cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, &source_size, &err); checkErr(err);
	std::string include_path = src_root + "/OpenCL/inc";
	if (include_path.find_first_of(" \t") != std::string::npos) include_path = "\"" + include_path + "\"";
	SetupBuildProgramWithSource(program, NULL, devices, include_path, "");
	return program;
}

// Written through "<name>.tmp-<host>-<pid>" + rename(), so concurrent engines never read a partial file.
static bool save_program_binary(cl_program program, const std::string& path) {
	cl_uint num_devices = 0;
	cl_int err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);
	if (err != CL_SUCCESS || num_devices == 0) return false;
	std::vector<size_t> sizes(num_devices, 0);
	err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * num_devices, sizes.data(), NULL);
	if (err != CL_SUCCESS || sizes[0] == 0) return false;
	std::vector<std::vector<unsigned char> > binaries(num_devices);
	std::vector<unsigned char*> pointers(num_devices);
	for (cl_uint i = 0; i < num_devices; i++) {
		binaries[i].resize(sizes[i] > 0 ? sizes[i] : 1);
		pointers[i] = binaries[i].data();
	}
	err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * num_devices, pointers.data(), NULL);
	if (err != CL_SUCCESS) return false;

	char host[256] = "host";
	gethostname(host, sizeof(host) - 1);
	host[sizeof(host) - 1] = '\0';
	const std::string tmp_path = path + ".tmp-" + host + "-" + std::to_string((long)getpid());
	FILE* fp = fopen(tmp_path.c_str(), "wb");
	if (fp == NULL) return false;
	bool ok = fwrite(pointers[0], 1, sizes[0], fp) == sizes[0];
	ok = (fclose(fp) == 0) && ok;
	ok = ok && std::rename(tmp_path.c_str(), path.c_str()) == 0;
	if (!ok) std::remove(tmp_path.c_str());
	return ok;
}

static void setup_kernel_programs(cl_program programs[2], cl_context context, cl_device_id* devices,
								  const std::string& binary_dir) {
	const std::string dir = binary_dir.empty() ? std::string(".") : binary_dir;
	const std::string paths[2] = { dir + "/" + kernel_binary_names[0], dir + "/" + kernel_binary_names[1] };
#ifdef BUILD_KERNEL_FROM_SOURCE
	const bool always_rebuild = true;
#else
	const bool always_rebuild = false;
#endif
	if (!always_rebuild && is_regular_file(paths[0]) && is_regular_file(paths[1])) {
		programs[0] = load_program_binary(context, devices, paths[0]);
		programs[1] = programs[0] != NULL ? load_program_binary(context, devices, paths[1]) : NULL;
		if (programs[0] != NULL && programs[1] != NULL) {
			printf("\nLoaded OpenCL kernels from %s", dir.c_str()); fflush(stdout);
			return;
		}
		if (programs[0] != NULL) clReleaseProgram(programs[0]);
		printf("\nCannot load the OpenCL kernels in %s (built for another device or driver?), rebuilding them",
			   dir.c_str()); fflush(stdout);
	}

	namespace fs = std::experimental::filesystem;
	const char* src_env = getenv("POCKETVINA_KERNEL_SRC");
	const std::string src_root = (src_env != NULL && src_env[0] != '\0') ? std::string(src_env) : std::string(".");
	// Never write into the working directory (an opencl_binary_path left at its default "." means "no cache") nor
	// into the kernel source tree, i.e. the package: old configs name its shipped QuickVina2-GPU-2-1 kernels.
	const char* not_saved = NULL;
	std::error_code ec, ec_cwd;
	const fs::path cwd = fs::current_path(ec_cwd);
	if (fs::equivalent(dir, cwd, ec_cwd))
		not_saved = "is the working directory";
	else if (is_inside(dir, src_root))
		not_saved = "is inside the kernel source directory";
	else
		fs::create_directories(dir, ec);
	bool saved = not_saved == NULL;

	const std::vector<std::string>* sources[2] = { &kernel1_sources, &kernel2_sources };
	for (int i = 0; i < 2; i++) {
		printf(i == 0 ? "\n\nBuild kernel %d from source" : "\nBuild kernel %d from source", i + 1); fflush(stdout);
		programs[i] = build_program_from_source(context, devices, src_root, *sources[i]);
		if (not_saved == NULL) saved = save_program_binary(programs[i], paths[i]) && saved;
	}
	if (not_saved != NULL)
		printf("\nOpenCL kernels not saved: opencl_binary_path %s", not_saved);
	else if (saved)
		printf("\nSaved OpenCL kernels to %s", dir.c_str());
	else
		printf("\nWarning: cannot save the OpenCL kernels to %s, they will be rebuilt next time", dir.c_str());
	fflush(stdout);
}

volatile enum { FINISH, DOCKING, ABORT} status;
void print_process() {
	int count = 0;
	printf("\n");
	do
	{
#ifdef WIN32
		Sleep(100);
#else
		sleep(1);
#endif
		printf("\rPerform docking|");
		for (int i = 0; i < count; i++)printf(" ");
		printf("=======");
		for (int i = 0; i < 30 - count; i++)printf(" ");
		printf("|"); fflush(stdout);

		count++;
		count %= 30;
	} while (status == DOCKING);
	if (status == FINISH) {
		printf("\rPerform docking|");
		for (int i = 0; i < 16; i++)printf("=");
		printf("done");
		for (int i = 0; i < 17; i++)printf("=");
		printf("|\n"); fflush(stdout);
	}
	else if (status == ABORT) {
		printf("\rPerform docking|");
		for (int i = 0; i < 16; i++)printf("=");
		printf("error");
		for (int i = 0; i < 16; i++)printf("=");
		printf("|\n"); fflush(stdout);
	}
}

std::vector<output_type> cl_to_vina(output_type_cl result_ptr[], 
									ligand_atom_coords_cl result_coords_ptr[],
									int thread, int lig_torsion_size) 
{
	std::vector<output_type> results_vina;
	int num_atoms;
	for (int i = 0; i < thread; i++) {
		output_type_cl tmp = result_ptr[i];
		ligand_atom_coords_cl tmp_coords = result_coords_ptr[i];
		conf tmp_c;
		tmp_c.ligands.resize(1);
		// Position
		for (int j = 0; j < 3; j++)tmp_c.ligands[0].rigid.position[j] = tmp.position[j];
		// Orientation
		qt q(tmp.orientation[0], tmp.orientation[1], tmp.orientation[2], tmp.orientation[3]);
		tmp_c.ligands[0].rigid.orientation = q;
		output_type tmp_vina(tmp_c, tmp.e);
		// torsion
		for (int j = 0; j < lig_torsion_size; j++)tmp_vina.c.ligands[0].torsions.push_back(tmp.lig_torsion[j]);
		// coords
		for (int j = 0; j < MAX_NUM_OF_ATOMS; j++) {
			vec v_tmp(tmp_coords.coords[j][0], tmp_coords.coords[j][1], tmp_coords.coords[j][2]);
			if ((v_tmp[0] != 0 || v_tmp[1] != 0) || (v_tmp[2] != 0)) tmp_vina.coords.push_back(v_tmp);
		}
		results_vina.push_back(tmp_vina);
		if (i == 0)num_atoms = tmp_vina.coords.size();
		if(num_atoms != tmp_vina.coords.size()){throw std::runtime_error("atom coords not match!");}
	}
	return results_vina;
}

void main_procedure_cl(cache& c, const std::vector<model>& ms,  const precalculate& p, const parallel_mc par,
	const vec& corner1, const vec& corner2, const int seed, std::vector<output_container>& outs, std::string opencl_binary_path,
	const std::vector<std::vector<std::string>> ligand_names, const int rilc_bfgs) {

/**************************************************************************/
/***************************    OpenCL Init    ****************************/
/**************************************************************************/

	cl_int err;
	cl_platform_id* platforms;
	cl_device_id* devices;
	cl_context context;
	cl_command_queue queue;
	cl_int gpu_platform_id = 0;
	SetupPlatform(&platforms, &gpu_platform_id);
	SetupDevice(platforms, &devices, gpu_platform_id);
	SetupContext(platforms, devices, &context, 1, gpu_platform_id);
	SetupQueue(&queue, context, devices);
	cl_program programs[2];

	//printf("\nSearch depth is set to %d", par.mc.search_depth);
	printf("\nUsing random seed: %d", seed);

	setup_kernel_programs(programs, context, devices, opencl_binary_path);

	err = clUnloadPlatformCompiler(platforms[gpu_platform_id]); checkErr(err);
	//Set kernel arguments
	cl_kernel kernels[2];
	char kernel_name[][50] = { "kernel1","kernel2"};
	SetupKernel(kernels, programs, 2, kernel_name);

	size_t max_wg_size; // max work item within one work group
	size_t max_wi_size[3]; // max work item within each dimension(global)
	err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL); checkErr(err);
	err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof(size_t), &max_wi_size, NULL); checkErr(err);

	/**************************************************************************/
	/************************    Original Vina code    ************************/
	/**************************************************************************/
	sz nat = num_atom_types(c.atu);

	szv needed;
	for (int i = 0; i < nat; i++) {
		if (!c.grids[i].initialized()) {
			needed.push_back(i);
			c.grids[i].init(c.gd);
		}
	}

	flv affinities(needed.size());                

	grid& g = c.grids[needed.front()];              

	const fl cutoff_sqr = p.cutoff_sqr();         

	grid_dims gd_reduced = szv_grid_dims(c.gd);     
	szv_grid ig(ms[0], gd_reduced, cutoff_sqr); // use ms[0]           
	//szv_grid ig2(ms[1], gd_reduced, cutoff_sqr); // use ms[0]  

	for (int i = 0; i < 3; i++) {
		if (ig.m_init[i] != g.m_init[i]) {
			printf("m_init not equal!");
			exit(-1);
		}
		if (ig.m_range[i] != g.m_range[i]) {
			printf("m_range not equal!");
			exit(-1);
		}
	}

	vec authentic_v(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl

	std::vector<conf_size> s;
	std::vector<output_type> tmps;
	for (int i = 0; i < ms.size(); i++) {
		s.push_back(ms[i].get_size());
		output_type tmp(s[i], 0);
		tmps.push_back(tmp);
	}
	//change g(s);
	
	//quasi_newton quasi_newton_par; const int quasi_newton_par_max_steps = par.mc.ssd_par.evals;
	/**************************************************************************/
	/************************    Kernel1    ***********************/
	/**************************************************************************/
	rng generator(static_cast<rng::result_type>(seed));
	model m = ms[0];

	// Preparing protein atoms related data
	pa_cl* pa_ptr = (pa_cl*)malloc(sizeof(pa_cl));
	if(MAX_NUM_OF_PROTEIN_ATOMS <= m.grid_atoms.size()){throw std::runtime_error("pocket too large!");}
	for (int i = 0; i < m.grid_atoms.size(); i++) {
		pa_ptr->atoms[i].types[0] = m.grid_atoms[i].el;
		pa_ptr->atoms[i].types[1] = m.grid_atoms[i].ad;
		pa_ptr->atoms[i].types[2] = m.grid_atoms[i].xs;
		pa_ptr->atoms[i].types[3] = m.grid_atoms[i].sy;
		for (int j = 0; j < 3; j++)pa_ptr->atoms[i].coords[j] = m.grid_atoms[i].coords.data[j];
	}
	size_t pa_size = sizeof(pa_cl);

	// Preaparing precalculated look up table related data
	pre_cl* pre_ptr = (pre_cl*)malloc(sizeof(pre_cl));
	pre_ptr->m_cutoff_sqr = p.cutoff_sqr();
	pre_ptr->factor = p.factor;
	pre_ptr->n = p.n;
	if(MAX_P_DATA_M_DATA_SIZE <= p.data.m_data.size()){throw std::runtime_error("LUT too large!");}
	for (int i = 0; i < p.data.m_data.size(); i++) {
		pre_ptr->m_data[i].factor = p.data.m_data[i].factor;
		if(FAST_SIZE != p.data.m_data[i].fast.size()){throw std::runtime_error("fast too large!");}
		if(SMOOTH_SIZE != p.data.m_data[i].smooth.size()){throw std::runtime_error("smooth too large!");}

		for (int j = 0; j < FAST_SIZE; j++) {
			pre_ptr->m_data[i].fast[j] = p.data.m_data[i].fast[j];
		}
		for (int j = 0; j < SMOOTH_SIZE; j++) {
			pre_ptr->m_data[i].smooth[j][0] = p.data.m_data[i].smooth[j].first;
			pre_ptr->m_data[i].smooth[j][1] = p.data.m_data[i].smooth[j].second;
		}
	}
	size_t pre_size = sizeof(pre_cl);

	// Preparing grid boundries
	gb_cl* gb_ptr = (gb_cl*)malloc(sizeof(gb_cl));
	gb_ptr->dims[0] = ig.m_data.dim0(); gb_ptr->dims[1] = ig.m_data.dim1(); gb_ptr->dims[2] = ig.m_data.dim2();
	for (int i = 0; i < 3; i++)gb_ptr->init[i] = ig.m_init.data[i];
	for (int i = 0; i < 3; i++)gb_ptr->range[i] = ig.m_range.data[i];
	size_t gb_size = sizeof(gb_cl);

	// Preparing atom relationship
	if(MAX_NUM_OF_ATOM_RELATION_COUNT < ig.m_data.m_data.size()){throw std::runtime_error("Relation too large! Define a large box (see readme) would help");}
	// assert(ig.m_data.m_i <= 10); assert(ig.m_data.m_j <= 10); assert(ig.m_data.m_k <= 10);
	ar_cl* ar_ptr = (ar_cl*)malloc(sizeof(ar_cl));
	for (int i = 0; i < ig.m_data.m_data.size(); i++) {
		ar_ptr->relation_size[i] = ig.m_data.m_data[i].size();
		if(MAX_NUM_OF_ATOM_RELATION_COUNT < ar_ptr->relation_size[i]){throw std::runtime_error("Relation too large! Define a large box (see readme) would help");}
		for (int j = 0; j < ar_ptr->relation_size[i]; j++) {
			ar_ptr->relation[i][j] = ig.m_data.m_data[i][j];
		}
	}
	size_t ar_size = sizeof(ar_cl);

	// Preparing grid related data
	if(GRIDS_SIZE != c.grids.size()){throw std::runtime_error("grid_size has to be 17!");} // grid_size has to be 17
	grids_cl* grids_ptr = (grids_cl*)malloc(sizeof(grids_cl)); if (grids_ptr == nullptr) {throw std::runtime_error("Grid too large! Define a large box (see readme) would help");}
	grid* tmp_grid_ptr = &c.grids[0];

	grids_ptr->atu = c.atu; // atu
	grids_ptr->slope = c.slope; // slope
	int grids_front;
	for (int i = GRIDS_SIZE - 1; i >= 0; i--) {
		for (int j = 0; j < 3; j++) {
			grids_ptr->grids[i].m_init[j] = tmp_grid_ptr[i].m_init[j];
			grids_ptr->grids[i].m_factor[j] = tmp_grid_ptr[i].m_factor[j];
			grids_ptr->grids[i].m_dim_fl_minus_1[j] = tmp_grid_ptr[i].m_dim_fl_minus_1[j];
			grids_ptr->grids[i].m_factor_inv[j] = tmp_grid_ptr[i].m_factor_inv[j];
		}
		if (tmp_grid_ptr[i].m_data.dim0() != 0) {
			grids_ptr->grids[i].m_i = tmp_grid_ptr[i].m_data.dim0(); if(MAX_NUM_OF_GRID_MI < grids_ptr->grids[i].m_i){throw std::runtime_error("MAX_NUM_OF_GRID_MI too small!  Define a large box (see readme) would help");}
			grids_ptr->grids[i].m_j = tmp_grid_ptr[i].m_data.dim1(); if(MAX_NUM_OF_GRID_MJ < grids_ptr->grids[i].m_j){throw std::runtime_error("MAX_NUM_OF_GRID_MJ too small!  Define a large box (see readme) would help");}
			grids_ptr->grids[i].m_k = tmp_grid_ptr[i].m_data.dim2(); if(MAX_NUM_OF_GRID_MK < grids_ptr->grids[i].m_k){throw std::runtime_error("MAX_NUM_OF_GRID_MK too small!  Define a large box (see readme) would help");}
			grids_front = i;
		}
		else {
			grids_ptr->grids[i].m_i = 0;
			grids_ptr->grids[i].m_j = 0;
			grids_ptr->grids[i].m_k = 0;
		}
	}
	size_t grids_size = sizeof(grids_cl);

	// miscellaneous
	mis_cl* mis_ptr = (mis_cl*)malloc(sizeof(mis_cl));
	mis_ptr->needed_size = needed.size();
	mis_ptr->epsilon_fl = epsilon_fl;
	mis_ptr->cutoff_sqr = cutoff_sqr;
	mis_ptr->max_fl = max_fl;
	//mis_ptr->torsion_size = tmp.c.ligands[0].torsions.size();
	//mis_ptr->max_bfgs_steps = quasi_newton_par_max_steps;
	//mis_ptr->search_depth = par.mc.search_depth;
	mis_ptr->mutation_amplitude = par.mc.mutation_amplitude;
	mis_ptr->total_wi = max_wi_size[0] * max_wi_size[1];
	mis_ptr->thread = par.mc.thread;
	mis_ptr->ar_mi = ig.m_data.m_i;
	mis_ptr->ar_mj = ig.m_data.m_j;
	mis_ptr->ar_mk = ig.m_data.m_k;
	mis_ptr->grids_front = grids_front;
	for (int i = 0; i < 3; i++) mis_ptr->authentic_v[i] = authentic_v[i];
	for (int i = 0; i < 3; i++) mis_ptr->hunt_cap[i] = par.mc.hunt_cap[i];
	size_t mis_size = sizeof(mis_cl);

	float* needed_ptr = (float*)malloc(mis_ptr->needed_size * sizeof(float));
	for (int i = 0; i < mis_ptr->needed_size; i++)needed_ptr[i] = needed[i];

	cl_mem pre_gpu;
	CreateDeviceBuffer(&pre_gpu, CL_MEM_READ_ONLY, pre_size, context);
	err = clEnqueueWriteBuffer(queue, pre_gpu, false, 0, pre_size, pre_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem pa_gpu;
	CreateDeviceBuffer(&pa_gpu, CL_MEM_READ_ONLY, pa_size, context);
	err = clEnqueueWriteBuffer(queue, pa_gpu, false, 0, pa_size, pa_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem gb_gpu;
	CreateDeviceBuffer(&gb_gpu, CL_MEM_READ_ONLY, gb_size, context);
	err = clEnqueueWriteBuffer(queue, gb_gpu, false, 0, gb_size, gb_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem ar_gpu;
	CreateDeviceBuffer(&ar_gpu, CL_MEM_READ_ONLY, ar_size, context);
	err = clEnqueueWriteBuffer(queue, ar_gpu, false, 0, ar_size, ar_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem grids_gpu;
	CreateDeviceBuffer(&grids_gpu, CL_MEM_READ_WRITE, grids_size, context);
	err = clEnqueueWriteBuffer(queue, grids_gpu, false, 0, grids_size, grids_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem needed_gpu;
	CreateDeviceBuffer(&needed_gpu, CL_MEM_READ_WRITE, mis_ptr->needed_size * sizeof(float), context);
	err = clEnqueueWriteBuffer(queue, needed_gpu, false, 0, mis_ptr->needed_size * sizeof(float), needed_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem mis_gpu;
	CreateDeviceBuffer(&mis_gpu, CL_MEM_READ_ONLY, mis_size, context);
	err = clEnqueueWriteBuffer(queue, mis_gpu, false, 0, mis_size, mis_ptr, 0, NULL, NULL); checkErr(err);
	
	clFinish(queue);

	SetKernelArg(kernels[0], 0, sizeof(cl_mem), &pre_gpu);
	SetKernelArg(kernels[0], 1, sizeof(cl_mem), &pa_gpu);
	SetKernelArg(kernels[0], 2, sizeof(cl_mem), &gb_gpu);
	SetKernelArg(kernels[0], 3, sizeof(cl_mem), &ar_gpu);
	SetKernelArg(kernels[0], 4, sizeof(cl_mem), &grids_gpu);
	SetKernelArg(kernels[0], 5, sizeof(cl_mem), &mis_gpu);
	SetKernelArg(kernels[0], 6, sizeof(cl_mem), &needed_gpu);
	SetKernelArg(kernels[0], 7, sizeof(int), &c.atu);
	SetKernelArg(kernels[0], 8, sizeof(int), &nat);

	status = DOCKING;
# ifdef NDEBUG
	std::thread console_thread(print_process);
# endif

	cl_event kernel1;
	size_t kernel1_global_size[3] = { 128, 128, 128 };
	size_t kernel1_local_size[3] = { 4,4,4 };

	err = clEnqueueNDRangeKernel(queue, kernels[0], 3, 0, kernel1_global_size, kernel1_local_size, 0, NULL, &kernel1); checkErr(err);
	clWaitForEvents(1, &kernel1);

	free(pa_ptr);
	free(gb_ptr);
	free(ar_ptr);
	free(needed_ptr);
	free(pre_ptr);
	free(grids_ptr);

	err = clReleaseMemObject(pa_gpu);		checkErr(err);
	err = clReleaseMemObject(gb_gpu);		checkErr(err);
	err = clReleaseMemObject(ar_gpu);		checkErr(err);
	err = clReleaseMemObject(needed_gpu);	checkErr(err);

	err = clReleaseEvent(kernel1);			checkErr(err);
/**************************************************************************/
/************************    Kernel2    ***********************/
/**************************************************************************/
	int num_ligands = ms.size();
	std::vector<random_maps*>			rand_maps_ptrs		(num_ligands);
	std::vector<output_type_cl*>		ric_ptrs			(num_ligands);
	std::vector<m_cl*>					m_ptrs				(num_ligands);
	std::vector<ligand_atom_coords_cl*> result_coords_ptrs	(num_ligands);
	std::vector<output_type_cl*>		result_ptrs			(num_ligands);
	std::vector<int>					torsion_sizes		(num_ligands);

	std::vector<cl_mem> ric_gpus			(num_ligands);
	std::vector<cl_mem> m_gpus				(num_ligands);
	std::vector<cl_mem> random_maps_gpus	(num_ligands);
	std::vector<cl_mem> result_coords_gpus	(num_ligands);
	std::vector<cl_mem> result_gpus			(num_ligands);

	std::vector<cl_event> ligands_events	(num_ligands);


	for (int ligand_count = 0; ligand_count < num_ligands; ligand_count++) {
# ifndef NDEBUG
		printf("\nDocking ligand: %s", ligand_names[ligand_count][0].c_str());
# endif
	try {
		model m = ms[ligand_count];
		if(m.atoms.size() >= MAX_NUM_OF_ATOMS){throw std::runtime_error("Ligand too large!");}
		output_type tmp = tmps[ligand_count];
		torsion_sizes[ligand_count] = tmp.c.ligands[0].torsions.size();
		if (tmp.c.ligands[0].torsions.size() >= MAX_NUM_OF_LIG_TORSION) { throw std::runtime_error("Ligand too large!"); }
		// Generate random maps
		rand_maps_ptrs[ligand_count] = (random_maps*)malloc(sizeof(random_maps));
		random_maps* rand_maps_ptr = rand_maps_ptrs[ligand_count];
		for (int i = 0; i < MAX_NUM_OF_RANDOM_MAP; i++) {
			rand_maps_ptr->int_map[i] = random_int(0, int(tmp.c.ligands[0].torsions.size()), generator);
			rand_maps_ptr->pi_map[i] = random_fl(-pi, pi, generator);
		}
		for (int i = 0; i < MAX_NUM_OF_RANDOM_MAP; i++) {
			vec rand_coords = random_inside_sphere(generator);
			for (int j = 0; j < 3; j++) {
				rand_maps_ptr->sphere_map[i][j] = rand_coords[j];
			}
		}
		size_t rand_maps_size = sizeof(*rand_maps_ptr);

		// Preparing random initial conformations
		ric_ptrs[ligand_count] = (output_type_cl*)malloc(par.mc.thread * sizeof(output_type_cl));
		output_type_cl* ric_ptr = ric_ptrs[ligand_count];
		for (int i = 0; i < par.mc.thread; i++) {
			tmp.c.randomize(corner1, corner2, generator);
			for (int j = 0; j < 3; j++)ric_ptr[i].position[j] = tmp.c.ligands[0].rigid.position[j];
			ric_ptr[i].orientation[0] = tmp.c.ligands[0].rigid.orientation.R_component_1();
			ric_ptr[i].orientation[1] = tmp.c.ligands[0].rigid.orientation.R_component_2();
			ric_ptr[i].orientation[2] = tmp.c.ligands[0].rigid.orientation.R_component_3();
			ric_ptr[i].orientation[3] = tmp.c.ligands[0].rigid.orientation.R_component_4();
			for (int j = 0; j < tmp.c.ligands[0].torsions.size(); j++)ric_ptr[i].lig_torsion[j] = tmp.c.ligands[0].torsions[j];
		}
		size_t ric_size = par.mc.thread * sizeof(output_type_cl);

		// Preparing m related data
		m_ptrs[ligand_count] = (m_cl*)malloc(sizeof(m_cl));
		m_cl* m_ptr = m_ptrs[ligand_count];
		for (int i = 0; i < m.atoms.size(); i++) {
			m_ptr->atoms[i].types[0] = m.atoms[i].el;// To store 4 atoms types (el, ad, xs, sy)
			m_ptr->atoms[i].types[1] = m.atoms[i].ad;
			m_ptr->atoms[i].types[2] = m.atoms[i].xs;
			m_ptr->atoms[i].types[3] = m.atoms[i].sy;
			for (int j = 0; j < 3; j++) {
				m_ptr->atoms[i].coords[j] = m.atoms[i].coords[j];// To store atom coords
			}
		}

		// To store atoms coords
		for (int i = 0; i < m.coords.size(); i++) {
			for (int j = 0; j < 3; j++) {
				m_ptr->m_coords.coords[i][j] = m.coords[i].data[j];
			}
		}

		//To store minus forces
		for (int i = 0; i < m.coords.size(); i++) {
			for (int j = 0; j < 3; j++) {
				m_ptr->minus_forces.coords[i][j] = m.minus_forces[i].data[j];
			}
		}

		// Preparing ligand data
		if(m.num_other_pairs() != 0){throw std::runtime_error("m.other_paris is not supported!");} // m.other_paris is not supported!
		if(m.ligands.size() != 1){throw std::runtime_error("Only one ligand supported!");} // Only one ligand supported!
		m_ptr->ligand.pairs.num_pairs = m.ligands[0].pairs.size();
		if (m.ligands[0].pairs.size() >= MAX_NUM_OF_LIG_PAIRS) { throw std::runtime_error("Ligand too large!"); }
		for (int i = 0; i < m_ptr->ligand.pairs.num_pairs; i++) {
			m_ptr->ligand.pairs.type_pair_index[i] = m.ligands[0].pairs[i].type_pair_index;
			m_ptr->ligand.pairs.a[i] = m.ligands[0].pairs[i].a;
			m_ptr->ligand.pairs.b[i] = m.ligands[0].pairs[i].b;
		}
		m_ptr->ligand.begin = m.ligands[0].begin; // 0
		m_ptr->ligand.end = m.ligands[0].end; // 29
		ligand m_ligand = m.ligands[0]; // Only support one ligand 
		if(m_ligand.end >= MAX_NUM_OF_ATOMS){throw std::runtime_error("Ligand too large!");}

		// Store root node
		m_ptr->ligand.rigid.atom_range[0][0] = m_ligand.node.begin;
		m_ptr->ligand.rigid.atom_range[0][1] = m_ligand.node.end;
		for (int i = 0; i < 3; i++) m_ptr->ligand.rigid.origin[0][i] = m_ligand.node.get_origin()[i];
		for (int i = 0; i < 9; i++) m_ptr->ligand.rigid.orientation_m[0][i] = m_ligand.node.get_orientation_m().data[i];
		m_ptr->ligand.rigid.orientation_q[0][0] = m_ligand.node.orientation().R_component_1();
		m_ptr->ligand.rigid.orientation_q[0][1] = m_ligand.node.orientation().R_component_2();
		m_ptr->ligand.rigid.orientation_q[0][2] = m_ligand.node.orientation().R_component_3();
		m_ptr->ligand.rigid.orientation_q[0][3] = m_ligand.node.orientation().R_component_4();
		for (int i = 0; i < 3; i++) { m_ptr->ligand.rigid.axis[0][i] = 0; m_ptr->ligand.rigid.relative_axis[0][i] = 0; m_ptr->ligand.rigid.relative_origin[0][i] = 0; }
		
		
		// Store children nodes (in depth-first order)
		struct tmp_struct {
			int start_index = 0;
			int parent_index = 0;
			void store_node(tree<segment>& child_ptr, rigid_cl& rigid) {
				start_index++; // start with index 1, index 0 is root node
				rigid.parent[start_index] = parent_index;
				rigid.atom_range[start_index][0] = child_ptr.node.begin;
				rigid.atom_range[start_index][1] = child_ptr.node.end;
				for (int i = 0; i < 9; i++) rigid.orientation_m[start_index][i] = child_ptr.node.get_orientation_m().data[i];
				rigid.orientation_q[start_index][0] = child_ptr.node.orientation().R_component_1();
				rigid.orientation_q[start_index][1] = child_ptr.node.orientation().R_component_2();
				rigid.orientation_q[start_index][2] = child_ptr.node.orientation().R_component_3();
				rigid.orientation_q[start_index][3] = child_ptr.node.orientation().R_component_4();
				for (int i = 0; i < 3; i++) {
					rigid.origin[start_index][i] = child_ptr.node.get_origin()[i];
					rigid.axis[start_index][i] = child_ptr.node.get_axis()[i];
					rigid.relative_axis[start_index][i] = child_ptr.node.relative_axis[i];
					rigid.relative_origin[start_index][i] = child_ptr.node.relative_origin[i];
				}

				if (child_ptr.children.size() == 0) return;
				else {	
					if (start_index >= MAX_NUM_OF_RIGID) { throw std::runtime_error("Children map too large!"); }
					int parent_index_tmp = start_index;
					for (int i = 0; i < child_ptr.children.size(); i++) {
						this->parent_index = parent_index_tmp; // Update parent index
						this->store_node(child_ptr.children[i], rigid);
					}
				}
			}
		};
		tmp_struct ts;
 		for (int i = 0; i < m_ligand.children.size(); i++) {
			ts.parent_index = 0; // Start a new branch, whose parent is 0
			ts.store_node(m_ligand.children[i], m_ptr->ligand.rigid);
		}
		m_ptr->ligand.rigid.num_children = ts.start_index;
		// set children_map
		for (int i = 0; i < MAX_NUM_OF_RIGID; i++)
			for (int j = 0; j < MAX_NUM_OF_RIGID; j++)
				m_ptr->ligand.rigid.children_map[i][j] = false;
		for (int i = 1; i < m_ptr->ligand.rigid.num_children + 1; i++) {
			int parent_index = m_ptr->ligand.rigid.parent[i];
			m_ptr->ligand.rigid.children_map[parent_index][i] = true;
		}
		m_ptr->m_num_movable_atoms = m.num_movable_atoms();
		size_t m_size = sizeof(m_cl);

		// Init ligand atom coords
		result_coords_ptrs[ligand_count] = (ligand_atom_coords_cl*)malloc(par.mc.thread * sizeof(ligand_atom_coords_cl));

		// Init results
		result_ptrs[ligand_count] = (output_type_cl*)malloc(par.mc.thread * sizeof(output_type_cl));


		//cl_mem ric_gpu;
		CreateDeviceBuffer(&ric_gpus[ligand_count], CL_MEM_READ_ONLY, ric_size, context);
		err = clEnqueueWriteBuffer(queue, ric_gpus[ligand_count], false, 0, ric_size, ric_ptr, 0, NULL, NULL); checkErr(err);

		//cl_mem m_gpu;
		CreateDeviceBuffer(&m_gpus[ligand_count], CL_MEM_READ_WRITE, m_size, context);
		err = clEnqueueWriteBuffer(queue, m_gpus[ligand_count], false, 0, m_size, m_ptr, 0, NULL, NULL); checkErr(err);


		//cl_mem random_maps_gpu;
		CreateDeviceBuffer(&random_maps_gpus[ligand_count], CL_MEM_READ_ONLY, rand_maps_size, context);
		err = clEnqueueWriteBuffer(queue, random_maps_gpus[ligand_count], false, 0, rand_maps_size, rand_maps_ptr, 0, NULL, NULL); checkErr(err);

		//cl_mem result_coords_gpu;
		CreateDeviceBuffer(&result_coords_gpus[ligand_count], CL_MEM_WRITE_ONLY, mis_ptr->thread * sizeof(ligand_atom_coords_cl), context);
		err = clEnqueueWriteBuffer(queue, result_coords_gpus[ligand_count], false, 0, mis_ptr->thread * sizeof(ligand_atom_coords_cl),
			result_coords_ptrs[ligand_count], 0, NULL, NULL); checkErr(err);

		//cl_mem results_gpu;
		CreateDeviceBuffer(&result_gpus[ligand_count], CL_MEM_WRITE_ONLY, par.mc.thread * sizeof(output_type_cl), context);
		err = clEnqueueWriteBuffer(queue, result_gpus[ligand_count], false, 0, par.mc.thread * sizeof(output_type_cl),
			result_ptrs[ligand_count], 0, NULL, NULL); checkErr(err);

		clFinish(queue);

		SetKernelArg(kernels[1], 0, sizeof(cl_mem), &ric_gpus[ligand_count]);
		SetKernelArg(kernels[1], 1, sizeof(cl_mem), &m_gpus[ligand_count]);
		SetKernelArg(kernels[1], 2, sizeof(cl_mem), &pre_gpu);
		SetKernelArg(kernels[1], 3, sizeof(cl_mem), &grids_gpu);
		SetKernelArg(kernels[1], 4, sizeof(cl_mem), &random_maps_gpus[ligand_count]);
		SetKernelArg(kernels[1], 5, sizeof(cl_mem), &result_coords_gpus[ligand_count]);
		SetKernelArg(kernels[1], 6, sizeof(cl_mem), &result_gpus[ligand_count]);
		SetKernelArg(kernels[1], 7, sizeof(cl_mem), &mis_gpu);
		SetKernelArg(kernels[1], 8, sizeof(int),	&torsion_sizes[ligand_count]);
		SetKernelArg(kernels[1], 9, sizeof(int),	&par.mc.search_depth[ligand_count]);
		SetKernelArg(kernels[1], 10, sizeof(int),	&par.mc.ssd_par.bfgs_steps[ligand_count]);
		SetKernelArg(kernels[1], 11, sizeof(int),	&rilc_bfgs);

		size_t kernel2_global_size[2] = { 512, 32 };
		size_t kernel2_local_size[2] = { 16,2 };

		err = clEnqueueNDRangeKernel(queue, kernels[1], 2, 0, kernel2_global_size, kernel2_local_size,
			0, NULL, &ligands_events[ligand_count]); checkErr(err);

		clWaitForEvents(1, &ligands_events[ligand_count]);
		
		err = clEnqueueReadBuffer(queue, result_gpus[ligand_count], false, 0, par.mc.thread * sizeof(output_type_cl),
			result_ptrs[ligand_count], 0, NULL, NULL); checkErr(err);

		err = clEnqueueReadBuffer(queue, result_coords_gpus[ligand_count], false, 0, par.mc.thread * sizeof(ligand_atom_coords_cl),
			result_coords_ptrs[ligand_count], 0, NULL, NULL); checkErr(err);

		clFinish(queue);

		std::vector<output_type> result_vina = cl_to_vina(result_ptrs[ligand_count], result_coords_ptrs[ligand_count],
			par.mc.thread, torsion_sizes[ligand_count]);
		// if empty, something goes wrong in the device part
		if (result_vina.size() == 0) {
			status = ABORT;
# ifdef NDEBUG
			console_thread.join();
# endif
			exit(-1);
		}

		// Write back to vina
		for (int i = 0; i < par.mc.thread; i++) {
			//assert(result_vina[i].coords.size()== mis_ptr->)
			add_to_output_container(outs[ligand_count], result_vina[i], par.mc.min_rmsd, par.mc.num_saved_mins);
		}
		// VINA_CHECK(!outs[ligand_count].empty());
		// VINA_CHECK(outs[ligand_count].front().e <= outs[ligand_count].back().e);

		free(rand_maps_ptrs[ligand_count]);
		free(ric_ptrs[ligand_count]);
		free(m_ptrs[ligand_count]);
		free(result_coords_ptrs[ligand_count]);
		free(result_ptrs[ligand_count]);



		err = clReleaseMemObject(result_coords_gpus[ligand_count]); checkErr(err);
		err = clReleaseMemObject(result_gpus[ligand_count]);		checkErr(err);
		err = clReleaseMemObject(ric_gpus[ligand_count]);			checkErr(err);
		err = clReleaseMemObject(m_gpus[ligand_count]);				checkErr(err);
		err = clReleaseMemObject(random_maps_gpus[ligand_count]);	checkErr(err);
		
#ifndef TIME_ANALYSIS
		err = clReleaseEvent(ligands_events[ligand_count]);			checkErr(err);
#endif // !TIME_ANALYSIS
	}
	catch(...){
		continue;
	}
	}

	free(mis_ptr);
	err = clReleaseMemObject(mis_gpu);		checkErr(err);
	err = clReleaseMemObject(grids_gpu);	checkErr(err);
	err = clReleaseMemObject(pre_gpu);		checkErr(err);
	status = FINISH;
# ifdef NDEBUG
	console_thread.join(); // wait the thread finish
# endif
#ifdef TIME_ANALYSIS
	// Output Analysis
	cl_ulong time_start, time_end;
	double total_time = 0;
	std::string log_file = "gpu_runtime.log";
	// delete file if file exists
	if (std::ifstream(log_file)) {
		std::remove(log_file.c_str());
	}
	for (int ligand_count = 0; ligand_count < ms.size(); ligand_count++) {
		cl_event event_tmp = ligands_events[ligand_count];
		err = clGetEventProfilingInfo(event_tmp, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); checkErr(err);
		err = clGetEventProfilingInfo(event_tmp, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); checkErr(err);
		total_time = time_end - time_start;
		err = clReleaseEvent(ligands_events[ligand_count]);			checkErr(err);
		printf("\nQuickVina 2-GPU3 ligand %d runtime = %0.3f s", ligand_count,(total_time / 1000000000.0));

		std::cout << std::setiosflags(std::ios::fixed);
		std::ofstream file(log_file, std::ios::app);
		if (file.is_open())
		{
			file << "QuickVina 2-GPU3 monte carlo runtime of ligand " << ligand_names[ligand_count][0] << " = " << std::setprecision(5) << (total_time / 1000000000.0) << " s" << std::endl;
			file.close();
		}
	}
	printf("\n");
#endif
#ifdef OPENCL_TIME_ANALYSIS
		cout << setiosflags(ios::fixed);
		std::ofstream file("gpu_runtime.log");
		if (file.is_open())
		{
			file << "GPU grid cache runtime = " << (kernel1_total_time / 1000000000.0) << " s" << std::endl;
			file << "GPU monte carlo runtime = " << (kernel2_total_time / 1000000000.0) << " s" << std::endl;
			file.close();
		}
#endif
}