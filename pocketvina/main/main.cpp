/*

   Copyright (c) 2006-2010, The Scripps Research Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Dr. Oleg Trott <ot14@columbia.edu>,
		   The Olson Lab,
		   The Scripps Research Institute

*/

#include <iostream>
#include <string>
#include <exception>
#include <vector> // ligand paths
#include <cmath> // for ceila
#include <boost/program_options.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/exception.hpp>
#include <boost/filesystem/convenience.hpp> // filesystem::basename
#include <boost/thread/thread.hpp> // hardware_concurrency // FIXME rm ?
#include "parse_pdbqt.h"
#include "parallel_mc.h"
#include "file.h"
#include "cache.h"
#include "non_cache.h"
#include "naive_non_cache.h"
#include "parse_error.h"
#include "everything.h"
#include "weighted_terms.h"
#include "current_weights.h"
#include "quasi_newton.h"
#include "tee.h"
#include "coords.h" // add_to_output_container
#include "main_procedure_cl.h"
#include <experimental/filesystem>
#include "../lib/json.hpp" // Taken from https://github.com/nlohmann/json/blob/develop/single_include/nlohmann/json.hpp

using boost::filesystem::path;

path make_path(const std::string& str) {
	//return path(str, boost::filesystem::native);
	return path(str);
}

void doing(int verbosity, const std::string& str, tee& log) {
	if (verbosity > 1) {
		log << str << std::string(" ... ");
		log.flush();
	}
}

void done(int verbosity, tee& log) {
	if (verbosity > 1) {
		log << "done.";
		log.endl();
	}
}
std::string default_output(const std::string& input_name) {
	std::string tmp = input_name;
	if (tmp.size() >= 6 && tmp.substr(tmp.size() - 6, 6) == ".pdbqt")
		tmp.resize(tmp.size() - 6); // FIXME?
	return tmp + "_out.pdbqt";
}

void write_all_output(model& m, const output_container& out, sz how_many,
	const std::string& output_name,
	const std::vector<std::string>& remarks) {
	if (out.size() < how_many)
		how_many = out.size();
	VINA_CHECK(how_many <= remarks.size());
	ofile f(make_path(output_name));
	VINA_FOR(i, how_many) {
		m.set(out[i].c);
		m.write_model(f, i + 1, remarks[i]); // so that model numbers start with 1
	}
}

void do_randomization(model& m,
	const std::string& out_name,
	const vec& corner1, const vec& corner2, int seed, int verbosity, tee& log) {
	conf init_conf = m.get_initial_conf();
	rng generator(static_cast<rng::result_type>(seed));
	if (verbosity > 1) {
		log << "Using random seed: " << seed;
		log.endl();
	}
	const sz attempts = 10000;
	conf best_conf = init_conf;
	fl best_clash_penalty = 0;
	VINA_FOR(i, attempts) {
		conf c = init_conf;
		c.randomize(corner1, corner2, generator);
		m.set(c);
		fl penalty = m.clash_penalty();
		if (i == 0 || penalty < best_clash_penalty) {
			best_conf = c;
			best_clash_penalty = penalty;
		}
	}
	m.set(best_conf);
	if (verbosity > 1) {
		log << "Clash penalty: " << best_clash_penalty; // FIXME rm?
		log.endl();
	}
	m.write_structure(make_path(out_name));
}

void refine_structure(model& m, const precalculate& prec, non_cache& nc, output_type& out, const vec& cap, sz max_steps = 1000) {
	change g(m.get_size());
	quasi_newton quasi_newton_par;
	quasi_newton_par.max_steps = max_steps;
	const fl slope_orig = nc.slope;
	VINA_FOR(p, 5) {
		nc.slope = 100 * std::pow(10.0, 2.0 * p);
		quasi_newton_par(m, prec, nc, out, g, cap);
		m.set(out.c); // just to be sure
		if (nc.within(m))
			break;
	}
	out.coords = m.get_heavy_atom_movable_coords();
	if (!nc.within(m))
		out.e = max_fl;
	nc.slope = slope_orig;
}

std::string vina_remark(fl e, fl lb, fl ub) {
	std::ostringstream remark;
	remark.setf(std::ios::fixed, std::ios::floatfield);
	remark.setf(std::ios::showpoint);
	remark << "REMARK VINA RESULT: "
		<< std::setw(9) << std::setprecision(1) << e
		<< "  " << std::setw(9) << std::setprecision(3) << lb
		<< "  " << std::setw(9) << std::setprecision(3) << ub
		<< '\n';
	return remark.str();
}

output_container remove_redundant(const output_container& in, fl min_rmsd) {
	output_container tmp;
	VINA_FOR_IN(i, in)
		add_to_output_container(tmp, in[i], min_rmsd, in.size());
	return tmp;
}

void do_search(model& m, const boost::optional<model>& ref, const scoring_function& sf, const precalculate& prec, const igrid& ig, const precalculate& prec_widened, const igrid& ig_widened, non_cache& nc, // nc.slope is changed
	const std::string& out_name,
	const vec& corner1, const vec& corner2,
	const parallel_mc& par, fl energy_range, sz num_modes,
	int seed, int verbosity, bool score_only, bool local_only, tee& log, const terms& t, const flv& weights) {
	conf_size s = m.get_size();
	conf c = m.get_initial_conf();
	fl e = max_fl;
	const vec authentic_v(1000, 1000, 1000);
	//score_only = true;
	if (score_only) {
		fl intramolecular_energy = m.eval_intramolecular(prec, authentic_v, c);
		naive_non_cache nnc(&prec); // for out of grid issues
		e = m.eval_adjusted(sf, prec, nnc, authentic_v, c, intramolecular_energy);
		log << "Affinity: " << std::fixed << std::setprecision(5) << e << " (kcal/mol)";
		log.endl();
		flv term_values = t.evale_robust(m);
		VINA_CHECK(term_values.size() == 5);
		log << "Intermolecular contributions to the terms, before weighting:\n";
		log << std::setprecision(5);
		log << "    gauss 1     : " << term_values[0] << '\n';
		log << "    gauss 2     : " << term_values[1] << '\n';
		log << "    repulsion   : " << term_values[2] << '\n';
		log << "    hydrophobic : " << term_values[3] << '\n';
		log << "    Hydrogen    : " << term_values[4] << '\n';
		VINA_CHECK(weights.size() == term_values.size() + 1);
		fl e2 = 0;
		VINA_FOR_IN(i, term_values)
			e2 += term_values[i] * weights[i];
		e2 = sf.conf_independent(m, e2);
		if (e < 100 && std::abs(e2 - e) > 0.05) {
			log << "WARNING: the individual terms are inconsisent with the\n";
			log << "WARNING: affinity. Consider reporting this as a bug:\n";
			log << "WARNING: http://vina.scripps.edu/manual.html#bugs\n";
		}
	}
	else if (local_only) {
		output_type out(c, e);
		doing(verbosity, "Performing local search", log);
		refine_structure(m, prec, nc, out, authentic_v, par.mc.ssd_par.evals);
		done(verbosity, log);
		fl intramolecular_energy = m.eval_intramolecular(prec, authentic_v, out.c);
		e = m.eval_adjusted(sf, prec, nc, authentic_v, out.c, intramolecular_energy);

		log << "Affinity: " << std::fixed << std::setprecision(5) << e << " (kcal/mol)";
		log.endl();
		if (!nc.within(m))
			log << "WARNING: not all movable atoms are within the search space\n";

		doing(verbosity, "Writing output", log);
		output_container out_cont;
		out_cont.push_back(new output_type(out));
		std::vector<std::string> remarks(1, vina_remark(e, 0, 0));
		write_all_output(m, out_cont, 1, out_name, remarks); // how_many == 1
		done(verbosity, log);
	}
	else {
		//seed = -226066464; printf("seed is fixed to -226066464!\n");
		rng generator(static_cast<rng::result_type>(seed));
		log << "Using random seed: " << seed;
		log.endl();
		output_container out_cont;
		//doing(verbosity, "Performing search", log);
		par(m, out_cont, prec, ig, prec_widened, ig_widened, corner1, corner2, generator);
		//done(verbosity, log);
		
		std::cout << std::endl;
		doing(verbosity, "Refining results", log);
		VINA_FOR_IN(i, out_cont)
			refine_structure(m, prec, nc, out_cont[i], authentic_v, par.mc.ssd_par.evals);

		if (!out_cont.empty()) {
			out_cont.sort();
			const fl best_mode_intramolecular_energy = m.eval_intramolecular(prec, authentic_v, out_cont[0].c);
			VINA_FOR_IN(i, out_cont)
				if (not_max(out_cont[i].e))
					out_cont[i].e = m.eval_adjusted(sf, prec, nc, authentic_v, out_cont[i].c, best_mode_intramolecular_energy);
			// the order must not change because of non-decreasing g (see paper), but we'll re-sort in case g is non strictly increasing
			out_cont.sort();
		}

		const fl out_min_rmsd = 1;
		out_cont = remove_redundant(out_cont, out_min_rmsd);

		done(verbosity, log);

		log.setf(std::ios::fixed, std::ios::floatfield);
		log.setf(std::ios::showpoint);
		log << '\n';
		log << "mode |   affinity | dist from best mode\n";
		log << "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n";
		log << "-----+------------+----------+----------\n";

		model best_mode_model = m;
		if (!out_cont.empty())
			best_mode_model.set(out_cont.front().c);

		sz how_many = 0;
		std::vector<std::string> remarks;
		VINA_FOR_IN(i, out_cont) {
			if (how_many >= num_modes || !not_max(out_cont[i].e) || out_cont[i].e > out_cont[0].e + energy_range) break; // check energy_range sanity FIXME
			++how_many;
			log << std::setw(4) << i + 1
				<< "    " << std::setw(9) << std::setprecision(1) << out_cont[i].e; // intermolecular_energies[i];
			m.set(out_cont[i].c);
			const model& r = ref ? ref.get() : best_mode_model;
			const fl lb = m.rmsd_lower_bound(r);
			const fl ub = m.rmsd_upper_bound(r);
			log << "  " << std::setw(9) << std::setprecision(3) << lb
				<< "  " << std::setw(9) << std::setprecision(3) << ub; // FIXME need user-readable error messages in case of failures

			remarks.push_back(vina_remark(out_cont[i].e, lb, ub));
			log.endl();
		}
		doing(verbosity, "Writing output", log);
		write_all_output(m, out_cont, how_many, out_name, remarks);
		done(verbosity, log);

		if (how_many < 1) {
			log << "WARNING: Could not find any conformations completely within the search space.\n"
				<< "WARNING: Check that it is large enough for all movable atoms, including those in the flexible side chains.";
			log.endl();
		}
	}
}

using json = nlohmann::json;
void main_procedure(std::vector<model>& ms, const boost::optional<model>& ref, // m is non-const (FIXME?)
	const std::vector<std::string>& out_names,
	bool score_only, bool local_only, bool randomize_only, bool no_cache,
	const grid_dims& gd, int exhaustiveness,
	const flv& weights,
	int cpu, int seed, int verbosity, sz num_modes, fl energy_range, 
	tee& log, int search_depth, int thread, std::string opencl_binary_path, std::vector<std::vector<std::string>> ligand_names, int rilc_bfgs,
	const std::string& receptor_file, int pocket_rank) {
	
	// Use pocket_rank as needed in your procedure
    std::cout << "Using pocket rank: " << pocket_rank << std::endl;

	doing(verbosity, "Setting up the scoring function", log);

	int ligand_num = ms.size();

	everything t;
	VINA_CHECK(weights.size() == 6);

	weighted_terms wt(&t, weights);
	precalculate prec(wt);
	const fl left = 0.25;
	const fl right = 0.25;
	precalculate prec_widened(prec); prec_widened.widen(left, right);

	done(verbosity, log);

	vec corner1(gd[0].begin, gd[1].begin, gd[2].begin);
	vec corner2(gd[0].end, gd[1].end, gd[2].end);

	parallel_mc par;

	if (search_depth != 0) {
		assert(search_depth >= 1);
		par.mc.search_depth.resize(ligand_num);
		par.mc.search_depth = std::vector<int>(ligand_num, search_depth);
		printf("Search_depth is fixed to %d\n", search_depth);
	}
	else {
		printf("Using heuristic search_depth\n");
		for (int ligand_count = 0; ligand_count < ligand_num; ligand_count++) {
			double tmp_steps;
			if(rilc_bfgs==1){tmp_steps = 1;}
			else{tmp_steps = 1;}
			if (tmp_steps < 1) tmp_steps = 1;
			par.mc.search_depth.push_back((int)tmp_steps);
		}
	}
	par.mc.thread = thread;

	for (int ligand_count = 0; ligand_count < ligand_num; ligand_count++) {
		par.mc.ssd_par.bfgs_steps.push_back(unsigned((25 + ms[ligand_count].num_movable_atoms()) / 3));
	}

	par.mc.min_rmsd = 1.0;
	par.mc.num_saved_mins = 20;
	par.mc.hunt_cap = vec(10, 10, 10);
	par.num_tasks = exhaustiveness;
	par.num_threads = cpu;
	par.display_progress = (verbosity > 1);

	const fl slope = 1e6; // FIXME: too large? used to be 100

	std::vector<non_cache> nc;

	for (int ligand_count = 0; ligand_count < ligand_num; ligand_count++) {
		non_cache tmp(ms[ligand_count], gd, &prec, slope);
		nc.push_back(tmp); // if gd has 0 n's, this will not constrain anything
	}

	doing(verbosity, "Analyzing the binding site", log);
	cache c("scoring_function_version001", gd, slope, atom_type::XS);
	done(verbosity, log);

	std::vector<output_container> out_conts(ligand_num);
	main_procedure_cl(c, ms, prec, par, corner1, corner2, seed, out_conts, opencl_binary_path, ligand_names, rilc_bfgs);
	
	const vec authentic_v(1000, 1000, 1000);
	const scoring_function& sf = wt;
	std::cout << std::endl;

	for (int ligand_count = 0; ligand_count < ligand_num; ligand_count++) {

		output_container out_cont = out_conts[ligand_count];
		model m = ms[ligand_count];

		std::string name_tmp = out_names[ligand_count];
		//int int1 = name_tmp.find("\\"); int int2 = name_tmp.find(".pdbqt");
		std::string name_tmp2 = name_tmp.substr(name_tmp.find("\\")+1, name_tmp.length() - name_tmp.find("\\") -11);
		std::cout << "Refining ligand " << name_tmp2 << " results...";
#ifndef NO_REFINEMENT
		VINA_FOR_IN(i, out_cont)
			refine_structure(m, prec, nc[ligand_count], out_cont[i], authentic_v, par.mc.ssd_par.bfgs_steps[ligand_count]);
#endif
		if (!out_cont.empty()) {
			out_cont.sort();
			const fl best_mode_intramolecular_energy = m.eval_intramolecular(prec, authentic_v, out_cont[0].c);
			VINA_FOR_IN(i, out_cont)
				if (not_max(out_cont[i].e))
					out_cont[i].e = m.eval_adjusted(sf, prec, nc[ligand_count], authentic_v, out_cont[i].c, best_mode_intramolecular_energy);
			// the order must not change because of non-decreasing g (see paper), but we'll re-sort in case g is non strictly increasing
			out_cont.sort();
		}
		else{
			std::cout << std::endl << "No results found for ligand " << name_tmp2 << std::endl << std::endl;
			continue;
		}

		const fl out_min_rmsd = 1;
		out_cont = remove_redundant(out_cont, out_min_rmsd);

		done(verbosity, log);

		log.setf(std::ios::fixed, std::ios::floatfield);
		log.setf(std::ios::showpoint);
		/*log << '\n';*/
		log << "mode |   affinity | dist from best mode\n";
		log << "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n";
		log << "-----+------------+----------+----------\n";

		model best_mode_model = m;
		if (!out_cont.empty())
			best_mode_model.set(out_cont.front().c);

		sz how_many = 0;
		std::vector<std::string> remarks;
		json output_json;
		std::string receptor_basename = boost::filesystem::basename(receptor_file);
		std::string ligand_basename = boost::filesystem::basename(out_names[ligand_count]);
		std::string ligand_directory = boost::filesystem::path(out_names[ligand_count]).parent_path().string();
		std::string output_prefix = ligand_directory + "/" + receptor_basename + "-pocket" + std::to_string(pocket_rank) + "-" + ligand_basename;
		std::string output_pdbqt = output_prefix + ".pdbqt";
        output_json[receptor_basename + "-pocket" + std::to_string(pocket_rank) + "-" + ligand_basename + "-" + "results"] = json::array();
		VINA_FOR_IN(i, out_cont) {
			if (how_many >= num_modes || !not_max(out_cont[i].e) || out_cont[i].e > out_cont[0].e + energy_range) break; // check energy_range sanity FIXME
			++how_many;
			log << std::setw(4) << i + 1
				<< "    " << std::setw(9) << std::setprecision(1) << out_cont[i].e; // intermolecular_energies[i];
			m.set(out_cont[i].c);
			const model& r = ref ? ref.get() : best_mode_model;
			const fl lb = m.rmsd_lower_bound(r);
			const fl ub = m.rmsd_upper_bound(r);
			log << "  " << std::setw(9) << std::setprecision(3) << lb
				<< "  " << std::setw(9) << std::setprecision(3) << ub; // FIXME need user-readable error messages in case of failures

			remarks.push_back(vina_remark(out_cont[i].e, lb, ub));
			log.endl();
			// Format the numbers for precision
			std::ostringstream affinity_stream, lb_stream, ub_stream;
			affinity_stream << std::fixed << std::setprecision(1) << out_cont[i].e;
			lb_stream << std::fixed << std::setprecision(3) << lb;
			ub_stream << std::fixed << std::setprecision(3) << ub;
			output_json[receptor_basename + "-pocket" + std::to_string(pocket_rank) + "-" + ligand_basename + "-" + "results"].push_back({
				{"mode", i + 1},
				{"affinity (kcal/mol)", std::stod(affinity_stream.str())},  // Convert back to double if necessary
				{"dist from rmsd l.b.", std::stod(lb_stream.str())},
				{"dist from rmsd u.b.", std::stod(ub_stream.str())},
				{"pocket rank", pocket_rank}
			});
		}
		std::cout << "Writing ligand " << output_pdbqt << " output...\n";
		write_all_output(m, out_cont, how_many, output_pdbqt, remarks);
		// Write JSON file
        std::string json_file = output_pdbqt.substr(0, output_pdbqt.find_last_of('.')) + ".json";
        std::ofstream json_out(json_file);
        if (json_out.is_open()) {
            json_out << output_json.dump(4);  // Pretty print with 4 spaces
            json_out.close();
            std::cout << "JSON file successfully saved to: " << json_file << std::endl;
        } else {
            std::cerr << "Failed to open JSON file for writing: " << json_file << std::endl;
        }

		if (how_many < 1) {
			log << "WARNING: Could not find any conformations completely within the search space.\n"
				<< "WARNING: Check that it is large enough for all movable atoms, including those in the flexible side chains.";
			log.endl();
		}
		done(verbosity, log);
		std::cout << std::endl;
	}
}

struct usage_error : public std::runtime_error {
	usage_error(const std::string& message) : std::runtime_error(message) {}
};

struct options_occurrence {
	bool some;
	bool all;
	options_occurrence() : some(false), all(true) {} // convenience
	options_occurrence& operator+=(const options_occurrence& x) {
		some = some || x.some;
		all = all && x.all;
		return *this;
	}
};

options_occurrence get_occurrence(boost::program_options::variables_map& vm, boost::program_options::options_description& d) {
	options_occurrence tmp;
	VINA_FOR_IN(i, d.options())
		if (vm.count((*d.options()[i]).long_name()))
			tmp.some = true;
		else
			tmp.all = false;
	return tmp;
}

void check_occurrence(boost::program_options::variables_map& vm, boost::program_options::options_description& d) {
	VINA_FOR_IN(i, d.options()) {
		const std::string& str = (*d.options()[i]).long_name();
		if (!vm.count(str))
			std::cerr << "Required parameter --" << str << " is missing!\n";
	}
}

model parse_bundle(const std::string& rigid_name, const boost::optional<std::string>& flex_name_opt, const std::vector<std::string>& ligand_names) {
	model tmp = (flex_name_opt) ? parse_receptor_pdbqt(make_path(rigid_name), make_path(flex_name_opt.get()))
		: parse_receptor_pdbqt(make_path(rigid_name));
	VINA_FOR_IN(i, ligand_names)
		tmp.append(parse_ligand_pdbqt(make_path(ligand_names[i])));
	return tmp;
}

model parse_bundle(const std::vector<std::string>& ligand_names) {
	VINA_CHECK(!ligand_names.empty()); // FIXME check elsewhere
	model tmp = parse_ligand_pdbqt(make_path(ligand_names[0]));
	VINA_RANGE(i, 1, ligand_names.size())
		tmp.append(parse_ligand_pdbqt(make_path(ligand_names[i])));
	return tmp;
}

model parse_bundle(const boost::optional<std::string>& rigid_name_opt, const boost::optional<std::string>& flex_name_opt, const std::vector<std::string>& ligand_names) {
	if (rigid_name_opt)
		return parse_bundle(rigid_name_opt.get(), flex_name_opt, ligand_names);
	else
		return parse_bundle(ligand_names);
}

int main(int argc, char* argv[]) {
	clock_t start = clock();

	using namespace boost::program_options;
	const std::string version_string = "QuickVina-GPU 2.1";
	const std::string error_message = "\n\n\
Please contact the author, Jiansheng Wu <jansen@njupt.edu>, so\n\
that this problem can be resolved. The reproducibility of the\n\
error may be vital, so please remember to include the following in\n\
your problem report:\n\
* the EXACT error message,\n\
* your version of the program,\n\
* the type of computer system you are running it on,\n\
* all command line options,\n\
* configuration file (if used),\n\
* ligand file as PDBQT,\n\
* receptor file as PDBQT,\n\
* flexible side chains file as PDBQT (if used),\n\
* output file as PDBQT (if any),\n\
* input (if possible),\n\
* random seed the program used (this is printed when the program starts).\n\
\n\
Thank you!\n";

	const std::string cite_message = "\
################################################################################\n\
# If you use PocketVina in your work, please cite:                             #\n\
#                                                                              #\n\
#   Sarigün, Ahmet, et al. “PocketVina Enables Scalable and Highly Accurate    #\n\
#   Physically Valid Docking through Multi-Pocket Conditioning.” arXiv (2025). #\n\
#                                                                              #\n\
# Please also cite the P2Rank and AutoDock Vina family papers. Their BibTeX    #\n\
# entries are available at:                                                    #\n\
#   https://github.com/BIMSBbioinfo/PocketVina#citations                       #\n\
#                                                                              #\n\
################################################################################\n";

	try {
		std::string rigid_name, ligand_name, flex_name, config_name, out_name, log_name;
		fl center_x = -8.654, center_y = 2.229, center_z = 19.715, size_x = 24.0, size_y = 26.25, size_z = 22.5;
		int cpu = 1, seed, exhaustiveness = 1, verbosity = 2, num_modes = 20;
		fl energy_range = 2.0;
		int search_depth = 0;
		int thread = 5000;
		std::string ligand_directory, output_directory, opencl_binary_path;
		opencl_binary_path = ".";
		int rilc_bfgs = 1;
		
		// -0.035579, -0.005156, 0.840245, -0.035069, -0.587439, 0.05846
		fl weight_gauss1 = -0.035579;
		fl weight_gauss2 = -0.005156;
		fl weight_repulsion = 0.840245;
		fl weight_hydrophobic = -0.035069;
		fl weight_hydrogen = -0.587439;
		fl weight_rot = 0.05846;
		bool score_only = false, local_only = false, randomize_only = false, help = false, help_advanced = false, version = false; // FIXME

		positional_options_description positional; // remains empty

		options_description inputs("Input");
		inputs.add_options()
			("receptor", value<std::string>(&rigid_name), "rigid part of the receptor (PDBQT)")
			("flex", value<std::string>(&flex_name), "flexible side chains, if any (PDBQT)")
			("ligand", value<std::string>(&ligand_name), "ligand (PDBQT)")
			("ligand_directory", value<std::string>(&ligand_directory), "ligand directory, if virtual screening is needed")
			("output_directory", value<std::string>(&output_directory), "output directory, if virtual screening is needed")
			("pocket_rank", value<int>()->default_value(1), "rank of the pocket to process") // New pocket_rank option
			("thread", value<int>(&thread), "the number of computing lanes in Vina-GPU")
			("search_depth", value<int>(&search_depth), "the number of search depth in monte carlo")
			("opencl_binary_path", value<std::string>(&opencl_binary_path)->default_value(opencl_binary_path), "opencl precompiled binary file path")
			("rilc_bfgs",value<int>(&rilc_bfgs)->default_value(rilc_bfgs), "rilc_bfgs enable or not")
			;
		//options_description search_area("Search area (required, except with --score_only)");
		options_description search_area("Search space (required)");
		search_area.add_options()
			("center_x", value<fl>(&center_x), "X coordinate of the center")
			("center_y", value<fl>(&center_y), "Y coordinate of the center")
			("center_z", value<fl>(&center_z), "Z coordinate of the center")
			("size_x", value<fl>(&size_x), "size in the X dimension (Angstroms)")
			("size_y", value<fl>(&size_y), "size in the Y dimension (Angstroms)")
			("size_z", value<fl>(&size_z), "size in the Z dimension (Angstroms)")

			;
		//options_description outputs("Output prefixes (optional - by default, input names are stripped of .pdbqt\nare used as prefixes. _001.pdbqt, _002.pdbqt, etc. are appended to the prefixes to produce the output names");
		options_description outputs("Output (optional)");
		outputs.add_options()
			("out", value<std::string>(&out_name), "output models (PDBQT), the default is chosen based on the ligand file name")
			("log", value<std::string>(&log_name), "optionally, write log file")
			;
		options_description advanced("Advanced options (see the manual)");
		advanced.add_options()
			//("score_only", bool_switch(&score_only), "score only - search space can be omitted")
			//("local_only", bool_switch(&local_only), "do local search only")
			("randomize_only", bool_switch(&randomize_only), "randomize input, attempting to avoid clashes")
			("weight_gauss1", value<fl>(&weight_gauss1)->default_value(weight_gauss1), "gauss_1 weight")
			("weight_gauss2", value<fl>(&weight_gauss2)->default_value(weight_gauss2), "gauss_2 weight")
			("weight_repulsion", value<fl>(&weight_repulsion)->default_value(weight_repulsion), "repulsion weight")
			("weight_hydrophobic", value<fl>(&weight_hydrophobic)->default_value(weight_hydrophobic), "hydrophobic weight")
			("weight_hydrogen", value<fl>(&weight_hydrogen)->default_value(weight_hydrogen), "Hydrogen bond weight")
			("weight_rot", value<fl>(&weight_rot)->default_value(weight_rot), "N_rot weight")
			;
		options_description misc("Misc (optional)");
		misc.add_options()
			//("cpu", value<int>(&cpu), "the number of CPUs to use (the default is to try to detect the number of CPUs or, failing that, use 1)")
			("seed", value<int>(&seed), "explicit random seed")
			//("exhaustiveness", value<int>(&exhaustiveness)->default_value(1), "exhaustiveness of the global search (roughly proportional to time): 1+")
			("num_modes", value<int>(&num_modes)->default_value(20), "maximum number of binding modes to generate")
			("energy_range", value<fl>(&energy_range)->default_value(20.0), "maximum energy difference between the best binding mode and the worst one displayed (kcal/mol)")
			;
		options_description config("Configuration file (optional)");
		config.add_options()
			("config", value<std::string>(&config_name), "the above options can be put here")
			;
		options_description info("Information (optional)");
		info.add_options()
			("help", bool_switch(&help), "display usage summary")
			("help_advanced", bool_switch(&help_advanced), "display usage summary with advanced options")
			("version", bool_switch(&version), "display program version")
			;
		options_description desc, desc_config, desc_simple;
		desc.add(inputs).add(search_area).add(outputs).add(advanced).add(misc).add(config).add(info);
		desc_config.add(inputs).add(search_area).add(outputs).add(advanced).add(misc);
		desc_simple.add(inputs).add(search_area).add(outputs).add(misc).add(config).add(info);

		variables_map vm;
		try {
			//store(parse_command_line(argc, argv, desc, command_line_style::default_style ^ command_line_style::allow_guessing), vm);
			store(command_line_parser(argc, argv)
				.options(desc)
				.style(command_line_style::default_style ^ command_line_style::allow_guessing)
				.positional(positional)
				.run(),
				vm);
			notify(vm);
		}
		catch (boost::program_options::error& e) {
			std::cerr << "Command line parse error: " << e.what() << '\n' << "\nCorrect usage:\n" << desc_simple << '\n';
			return 1;
		}
		if (vm.count("config")) {
			try {
				path name = make_path(config_name);
				ifile config_stream(name);
				store(parse_config_file(config_stream, desc_config), vm);
				notify(vm);
			}
			catch (boost::program_options::error& e) {
				std::cerr << "Configuration file parse error: " << e.what() << '\n' << "\nCorrect usage:\n" << desc_simple << '\n';
				return 1;
			}
		}
		int pocket_rank = 1; // Default value
		if (vm.count("pocket_rank")) {
			pocket_rank = vm["pocket_rank"].as<int>();
			std::cout << "Pocket rank: " << pocket_rank << std::endl;
		}
		if (help) {
			std::cout << desc_simple << '\n';
			return 0;
		}
		if (help_advanced) {
			std::cout << desc << '\n';
			return 0;
		}
		if (version) {
			std::cout << version_string << '\n';
			return 0;
		}

		bool search_box_needed = !score_only; // randomize_only and local_only still need the search space
		bool output_produced = !score_only;
		bool receptor_needed = !randomize_only;

		if (receptor_needed) {
			if (vm.count("receptor") <= 0) {
				std::cerr << "Missing receptor.\n" << "\nCorrect usage:\n" << desc_simple << '\n';
				return 1;
			}
		}
		/*if (vm.count("ligand") <= 0) {
			std::cerr << "Missing ligand.\n" << "\nCorrect usage:\n" << desc_simple << '\n';
			return 1;
		}*/
		if (vm.count("thread") <= 0) {
			std::cerr << "Missing thread, the recommend thread value is 8000";
			return 1;
		}else if (thread < 1000) {
			throw usage_error("thread must be at least 1000");
		}

		if (vm.count("ligand") > 0 && vm.count("ligand_directory") == 1) {
			std::cerr << "Can not define ligand and ligand_directory at the same time\n";
			return 1;
		}

		if (cpu < 1)
			cpu = 1;
		if (vm.count("seed") == 0)
			seed = auto_seed();
		if (exhaustiveness < 1)
			throw usage_error("exhaustiveness must be 1 or greater");
		if (num_modes < 1)
			throw usage_error("num_modes must be 1 or greater");
		sz max_modes_sz = static_cast<sz>(num_modes);

		boost::optional<std::string> rigid_name_opt;
		if (vm.count("receptor"))
			rigid_name_opt = rigid_name;

		boost::optional<std::string> flex_name_opt;
		if (vm.count("flex"))
			flex_name_opt = flex_name;

		if (vm.count("flex") && !vm.count("receptor"))
			throw usage_error("Flexible side chains are not allowed without the rest of the receptor"); // that's the only way parsing works, actually

		tee log;
		if (vm.count("log") > 0)
			log.init(log_name);

		if (search_box_needed) {
			options_occurrence oo = get_occurrence(vm, search_area);
			if (!oo.all) {
				check_occurrence(vm, search_area);
				std::cerr << "\nCorrect usage:\n" << desc_simple << std::endl;
				return 1;
			}
			if (size_x <= 0 || size_y <= 0 || size_z <= 0)
				throw usage_error("Search space dimensions should be positive");
		}

		log << cite_message << '\n';

		if (search_box_needed && size_x * size_y * size_z > 27e3) {
			log << "WARNING: The search space volume > 27000 Angstrom^3 (See FAQ)\n";
		}

		if (vm.count("ligand") <= 0 && vm.count("ligand_directory") == 1) {
			std::cout << "Using virtual sreening mode\n\n";
		}

		if (vm.count("ligand") <= 0 && vm.count("ligand_directory") <= 0) {
			std::cerr << "Missing ligand\n";
			return 1;
		}

		if (vm.count("ligand") > 0 && vm.count("ligand_directory") <= 0) {
			std::cout << "Using single ligand docking mode\n\n";
			if (output_produced) { // FIXME
				if (!vm.count("out")) {
					out_name = default_output(ligand_name);
					std::cout << "Output will be " << out_name << '\n';
				}
			}
		}

		if (vm.count("search_depth") == 1 && search_depth < 1) {
			throw usage_error("search_depth must be 1 or greater");
		}

		grid_dims gd; // n's = 0 via default c'tor

		flv weights;
		weights.push_back(weight_gauss1);
		weights.push_back(weight_gauss2);
		weights.push_back(weight_repulsion);
		weights.push_back(weight_hydrophobic);
		weights.push_back(weight_hydrogen);
		weights.push_back(5 * weight_rot / 0.1 - 1); // linearly maps onto a different range, internally. see everything.cpp

		if (search_box_needed) {
			const fl granularity = 0.375;
			vec span(size_x, size_y, size_z);
			vec center(center_x, center_y, center_z);
			VINA_FOR_IN(i, gd) {
				gd[i].n = sz(std::ceil(span[i] / granularity));
				fl real_span = granularity * gd[i].n;
				gd[i].begin = center[i] - real_span / 2;
				gd[i].end = gd[i].begin + real_span;
			}
		}
		//if (vm.count("cpu") == 0) {
		//	unsigned num_cpus = boost::thread::hardware_concurrency();
		//	if (verbosity > 1) {
		//		if (num_cpus > 0)
		//			log << "Detected " << num_cpus << " CPU" << ((num_cpus > 1) ? "s" : "") << '\n';
		//		else
		//			log << "Could not detect the number of CPUs, using 1\n";
		//	}
		//	if (num_cpus > 0)
		//		cpu = num_cpus;
		//	else
		//		cpu = 1;
		//}
		if (cpu < 1)
			cpu = 1;
		if (verbosity > 1 && exhaustiveness < cpu)
			log << "WARNING: at low exhaustiveness, it may be impossible to utilize all CPUs\n";

		

		std::vector<std::vector<std::string>> ligand_names;
		std::vector<std::string> out_names;
		
		if (vm.count("ligand_directory") == 1) {
			std::string out_dir;
			if (vm.count("output_directory") == 1)
				out_dir = output_directory;
			else
				out_dir = ligand_directory + "_out";

			std::experimental::filesystem::create_directory(out_dir);
			for (const auto& entry : std::experimental::filesystem::directory_iterator(ligand_directory)) {
				std::vector<std::string> tmp = { entry.path().string() };
				ligand_names.push_back(tmp);
				std::string delimiter1 = ".pdbqt";
				std::string delimiter2 = ligand_directory;
				std::string tmp2 = entry.path().string();
				std::string tmp3 = tmp2.substr(0, tmp2.find(delimiter1)) + "_out.pdbqt";
				std::string tmp4 = out_dir + tmp3.substr(ligand_directory.length(), tmp3.length());
				out_names.push_back(tmp4);
			}
			std::cout << "Output will be in the directory " << out_dir << std::endl;
		}
		else {
			std::vector<std::string> tmp = { ligand_name };
			ligand_names.push_back(tmp);
			std::vector<std::string> tmp2 = { out_name };
			out_names.push_back(out_name);
		}

		doing(verbosity, "Reading input", log);
		std::vector<model> ms;
		std::vector<std::string> out_names_valid;
		for (int i = 0; i < ligand_names.size(); i++) {
			try {
				ms.push_back(parse_bundle(rigid_name_opt, flex_name_opt, ligand_names[i]));
				out_names_valid.push_back(out_names[i]);
			}
			catch (parse_error& e) {
				std::cerr << "\nParse error on line " << e.line << " in file \"" << e.file.string() << "\": " << e.reason << '\n';
				continue;
			}
		}

		boost::optional<model> ref;
		done(verbosity, log);
		if (ms.size() == 0) {
			std::cerr << "No valid ligands in the input directory\n";
			return 1;
		}

		main_procedure(ms, ref,
			out_names_valid,
			score_only, local_only, randomize_only, false, // no_cache == false
			gd, exhaustiveness,
			weights,
			cpu, seed, verbosity, max_modes_sz, energy_range, log, search_depth, thread, opencl_binary_path, ligand_names, rilc_bfgs, rigid_name, pocket_rank);
	}
	catch (file_error& e) {
		std::cerr << "\n\nError: could not open \"" << e.name.string() << "\" for " << (e.in ? "reading" : "writing") << ".\n";
		return 1;
	}
	catch (boost::filesystem::filesystem_error& e) {
		std::cerr << "\n\nFile system error: " << e.what() << '\n';
		return 1;
	}
	catch (usage_error& e) {
		std::cerr << "\n\nUsage error: " << e.what() << ".\n";
		return 1;
	}
	catch (parse_error& e) {
		std::cerr << "\n\nParse error on line " << e.line << " in file \"" << e.file.string() << "\": " << e.reason << '\n';
		return 1;
	}
	catch (std::bad_alloc&) {
		std::cerr << "\n\nError: insufficient memory!\n";
		return 1;
	}

	// Errors that shouldn't happen:

	catch (std::exception& e) {
		std::cerr << "\n\nAn error occurred: " << e.what() << ". " << error_message;
		return 1;
	}
	catch (internal_error& e) {
		std::cerr << "\n\nAn internal error occurred in " << e.file << "(" << e.line << "). " << error_message;
		return 1;
	}
	catch (...) {
		std::cerr << "\n\nAn unknown error occurred. " << error_message;
		return 1;
	}

	clock_t end = clock();

#ifdef TIME_ANALYSIS
	std::ofstream file("gpu_runtime.log", std::ios::app);
	if (file.is_open())
	{
		file << "QuickVina 2-GPU3 total runtime = " << (double)(end - start) / CLOCKS_PER_SEC << " s" << std::endl;
		file.close();
	}
#endif

	std::cout << "QuickVina 2-GPU3 total runtime = " << (double)(end - start) / CLOCKS_PER_SEC << " s" << std::endl;
	//getchar();
}
