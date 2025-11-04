from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
import pandas as pd
import logging
from pathlib import Path
from pocketvina import PocketVinaGPU, PocketVinaConfig, P2RankConfig, PocketConfig
from typing import Optional
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global variable to store uploaded protein files
uploaded_proteins = []  # List of dicts: {'original_name': 'protein.pdb', 'file_path': '/path/to/file', 'filename': 'timestamp_protein.pdb'}

# Global variable to store processed visualization files after docking
visualization_files = {}  # Dict: {'protein_name': {'processed_pdb': 'path', 'chimerax_script': 'path', 'pymol_script': 'path'}}

def save_uploaded_file(file):
    """Save uploaded file and return its path"""
    # Create a persistent upload directory
    upload_dir = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Create unique filename to avoid conflicts
    import time
    timestamp = str(int(time.time()))
    filename = f"{timestamp}_{file.filename}"
    save_path = os.path.join(upload_dir, filename)
    
    file.save(save_path)
    logger.info(f"Saved file to {save_path}")
    return os.path.abspath(save_path)

def create_protein_files_txt(output_dir):
    """Create protein_files.txt from uploaded proteins"""
    global uploaded_proteins
    
    if not uploaded_proteins:
        return None
        
    protein_files_path = os.path.join(output_dir, "protein_files.txt")
    with open(protein_files_path, "w") as f:
        for protein_info in uploaded_proteins:
            f.write(protein_info['file_path'] + "\n")
    
    logger.info(f"Created protein_files.txt with {len(uploaded_proteins)} proteins")
    return protein_files_path

def create_sortable_table(df, output_dir: Optional[str] = None):
    """Create a sortable HTML table from DataFrame"""
    if df.empty:
        return '<p class="text-muted">No results to display</p>'
    
    # Start table HTML
    html = ['<table class="table table-striped table-hover" id="results-table">']
    
    # Create header with sortable buttons
    html.append('<thead class="table-dark">')
    html.append('<tr>')
    for i, col in enumerate(df.columns):
        html.append(f'''
            <th class="sortable-header" onclick="sortTable({i})">
                <div class="d-flex align-items-center justify-content-between">
                    <span>{col}</span>
                    <div class="sort-buttons">
                        <i class="fas fa-sort sort-icon" id="sort-icon-{i}"></i>
                    </div>
                </div>
            </th>
        ''')
    # Extra column for inline 3D viewer toggle
    html.append('<th>3D</th>')
    html.append('</tr>')
    html.append('</thead>')
    
    # Create body
    html.append('<tbody>')
    row_counter = 0
    for _, row in df.iterrows():
        html.append('<tr>')
        # Pre-compute protein and ligand values for inline viewer
        protein_value = None
        for pcol, pval in row.items():
            if pcol.lower() in ['protein', 'protein_file', 'protein_path']:
                protein_value = str(pval) if not pd.isna(pval) else None
                break

        ligand_value = None
        for lcol, lval in row.items():
            if lcol.lower() in ['ligand', 'ligand_file', 'output_file', 'pdbqt_file', 'result_file']:
                ligand_value = str(lval) if not pd.isna(lval) else None
                break
        # Try to derive ligand path if missing using molecule_name and output_dir search
        if not ligand_value and output_dir:
            # molecule name column candidates
            mol_name = None
            for mcol, mval in row.items():
                if mcol.lower() in ['molecule_name', 'ligand_name']:
                    mol_name = str(mval) if not pd.isna(mval) else None
                    break
            if mol_name and protein_value:
                try:
                    protein_base = (protein_value.split('/')[-1] if '/' in protein_value else protein_value)
                    protein_base = protein_base.replace('.pdb', '')
                    # search batch_*_out for files containing protein_base and mol_name
                    found_path = None
                    for item in os.listdir(output_dir):
                        item_path = os.path.join(output_dir, item)
                        if os.path.isdir(item_path) and item.startswith('batch_') and item.endswith('_out'):
                            for root, dirs, files in os.walk(item_path):
                                for f in files:
                                    if not f.lower().endswith('.pdbqt'):
                                        continue
                                    name_l = f.lower()
                                    if protein_base.lower() in name_l and mol_name.lower().replace(' ', '_') in name_l:
                                        found_path = os.path.join(root, f)
                                        break
                                if found_path:
                                    break
                        if found_path:
                            break
                    if found_path:
                        ligand_value = found_path
                except Exception as _e:
                    pass

        processed_protein_file = None
        prot_filename = None
        if protein_value:
            prot_filename = protein_value.split('/')[-1] if '/' in protein_value else protein_value
            for base_name, files in visualization_files.items():
                if base_name in prot_filename or prot_filename.replace('.pdb', '').replace('-pocket1', '').replace('-pocket2', '').replace('-pocket3', '') in base_name:
                    processed_protein_file = files.get('processed_pdb')
                    break
        for col_idx, (col_name, value) in enumerate(row.items()):
            # Format numeric values
            if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}" if abs(value) < 1000 else f"{value:.2e}"
                else:
                    formatted_value = str(value)
            else:
                formatted_value = str(value) if not pd.isna(value) else "N/A"
            
            # Add 3D view button for certain columns
            if col_name.lower() in ['protein', 'protein_file', 'protein_path'] and formatted_value != "N/A":
                # For Mol* integration, we only display the protein name here.
                # The actual 3D preview (protein + ligand together) will be provided from the ligand column.
                filename = formatted_value.split('/')[-1] if '/' in formatted_value else formatted_value
                html.append(f'''<td>{filename}</td>''')
            elif col_name.lower() in ['ligand', 'ligand_file', 'output_file', 'pdbqt_file'] and formatted_value != "N/A":
                # Get corresponding protein file for complex view
                protein_col_name = None
                protein_value = None
                for pcol, pval in row.items():
                    if pcol.lower() in ['protein', 'protein_file', 'protein_path']:
                        protein_col_name = pcol
                        protein_value = str(pval) if not pd.isna(pval) else "N/A"
                        break
                
                # Find actual processed protein filename
                processed_protein_file = None
                chimerax_script = None
                pymol_script = None
                if protein_value and protein_value != "N/A":
                    prot_filename = protein_value.split('/')[-1] if '/' in protein_value else protein_value
                    
                    # Look for processed protein file
                    for base_name, files in visualization_files.items():
                        if base_name in prot_filename or prot_filename.replace('.pdb', '').replace('-pocket1', '').replace('-pocket2', '').replace('-pocket3', '') in base_name:
                            processed_protein_file = files.get('processed_pdb')
                            chimerax_script = files.get('chimerax_script')
                            pymol_script = files.get('pymol_script')
                            break
                
                if processed_protein_file:
                    # Single Mol* preview button: open in-app Mol* preview with both URLs
                    html.append(f'''
                        <td>
                            <div class="d-flex align-items-center justify-content-between">
                                <span>{formatted_value}</span>
                                <button class="btn btn-outline-primary view-3d-btn btn-sm" 
                                        title="View protein + ligand in Mol* (processed protein)"
                                        onclick="(function(){{
                                            const origin = window.location.origin;
                                            const p = encodeURIComponent(origin + '/view_processed_file?path=' + encodeURIComponent('{processed_protein_file}'));
                                            const l = encodeURIComponent(origin + '/view_result_file?path=' + encodeURIComponent('{formatted_value}'));
                                            openMolstarOverlay(origin + '/molstar_preview?protein-url=' + p + '&ligand-url=' + l);
                                        }})()">
                                    <i class="fas fa-cube"></i>
                                </button>
                            </div>
                        </td>
                    ''')
                else:
                    # Fallback: use original uploaded protein with ligand in in-app Mol* preview
                    html.append(f'''
                        <td>
                            <div class="d-flex align-items-center justify-content-between">
                                <span>{formatted_value}</span>
                                <button class="btn btn-outline-primary view-3d-btn btn-sm" 
                                        title="View protein + ligand in Mol* (original protein)"
                                        onclick="(function(){{
                                            const origin = window.location.origin;
                                            const p = encodeURIComponent(origin + '/view_protein/{prot_filename}');
                                            const l = encodeURIComponent(origin + '/view_result_file?path=' + encodeURIComponent('{formatted_value}'));
                                            openMolstarOverlay(origin + '/molstar_preview?protein-url=' + p + '&ligand-url=' + l);
                                        }})()">
                                    <i class="fas fa-cube"></i>
                                </button>
                            </div>
                        </td>
                    ''')
            else:
                html.append(f'<td>{formatted_value}</td>')
        # Inline 3D column
        protein_route = None
        ligand_route = None
        if ligand_value:
            ligand_route = f"/view_result_file?path={ligand_value}"
        if processed_protein_file:
            protein_route = f"/view_processed_file?path={processed_protein_file}"
        elif prot_filename:
            protein_route = f"/view_protein/{prot_filename}"

        if protein_route and ligand_route:
            html.append(f'''<td>
                <button class="btn btn-outline-primary btn-sm" title="3D Viewer"
                    onclick="(function(){{
                        const origin = window.location.origin;
                        const p = encodeURIComponent(origin + '{protein_route}');
                        const l = encodeURIComponent(origin + '{ligand_route}');
                        openMolstarOverlay(origin + '/molstar_preview?protein-url=' + p + '&ligand-url=' + l);
                    }})()">
                    <i class="fas fa-cube"></i> 3D Viewer
                </button>
            </td>''')
        else:
            html.append('<td><span class="text-muted">N/A</span></td>')
        html.append('</tr>')
        row_counter += 1
    html.append('</tbody>')
    html.append('</table>')
    
    return '\n'.join(html)

def scan_visualization_files(output_dir):
    """Scan for processed visualization files after docking"""
    global visualization_files
    visualization_files = {}
    
    vis_dir = os.path.join(output_dir, "visualizations")
    if not os.path.exists(vis_dir):
        logger.warning(f"Visualization directory not found: {vis_dir}")
        return
    
    logger.info(f"Scanning visualization files in: {vis_dir}")
    
    # Scan all files in visualization directory
    for filename in os.listdir(vis_dir):
        filepath = os.path.join(vis_dir, filename)
        
        # Extract protein base name
        if filename.endswith('_processed.pdb'):
            base_name = filename.replace('_processed.pdb', '')
            if base_name not in visualization_files:
                visualization_files[base_name] = {}
            visualization_files[base_name]['processed_pdb'] = filepath
            logger.info(f"Found processed PDB: {base_name} -> {filepath}")
            
        elif filename.endswith('_processed.pdb_chimerax.cxc'):
            base_name = filename.replace('_processed.pdb_chimerax.cxc', '')
            if base_name not in visualization_files:
                visualization_files[base_name] = {}
            visualization_files[base_name]['chimerax_script'] = filepath
            logger.info(f"Found ChimeraX script: {base_name} -> {filepath}")
            
        elif filename.endswith('_processed.pdb_pymol.pml'):
            base_name = filename.replace('_processed.pdb_pymol.pml', '')
            if base_name not in visualization_files:
                visualization_files[base_name] = {}
            visualization_files[base_name]['pymol_script'] = filepath
            logger.info(f"Found PyMOL script: {base_name} -> {filepath}")
    
    logger.info(f"Total visualization entries found: {len(visualization_files)}")
    for name, files in visualization_files.items():
        logger.info(f"  {name}: {list(files.keys())}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/molstar_preview')
def molstar_preview():
    # Standalone Mol* preview for the main app
    return render_template('molstar_preview.html')

@app.route('/add_protein', methods=['POST'])
def add_protein():
    global uploaded_proteins
    
    if 'protein_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['protein_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.pdb'):
        return jsonify({'error': 'Only PDB files are allowed'}), 400
    
    # Save the file
    try:
        file_path = save_uploaded_file(file)
        
        # Store protein info as dictionary
        protein_info = {
            'original_name': file.filename,
            'file_path': file_path,
            'filename': os.path.basename(file_path)
        }
        uploaded_proteins.append(protein_info)
        logger.info(f"Added protein file: {file.filename} -> {file_path}")
        
        return jsonify({
            'success': True,
            'message': f'Added {file.filename}. Total proteins: {len(uploaded_proteins)}',
            'proteins': [p['original_name'] for p in uploaded_proteins],
            'protein_info': uploaded_proteins  # Send full info for 3D viewer
        })
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return jsonify({'error': f'Error saving file: {str(e)}'}), 500

@app.route('/clear_proteins', methods=['POST'])
def clear_proteins():
    global uploaded_proteins
    
    # Clean up uploaded files
    for protein_info in uploaded_proteins:
        try:
            if os.path.exists(protein_info['file_path']):
                os.remove(protein_info['file_path'])
                logger.info(f"Deleted file: {protein_info['file_path']}")
        except Exception as e:
            logger.warning(f"Could not delete file {protein_info['file_path']}: {e}")
    
    uploaded_proteins = []
    return jsonify({
        'success': True,
        'message': 'All proteins cleared',
        'proteins': [],
        'protein_info': []
    })

@app.route('/run_docking', methods=['POST'])
def run_docking():
    global uploaded_proteins
    
    try:
        # Check if ligands file is uploaded
        if 'ligands_file' not in request.files:
            return jsonify({'error': 'No ligands file uploaded'}), 400
        
        ligands_file = request.files['ligands_file']
        if ligands_file.filename == '':
            return jsonify({'error': 'No ligands file selected'}), 400
        
        # Check if proteins are uploaded
        if not uploaded_proteins:
            return jsonify({'error': 'Please upload at least one protein file'}), 400
        
        # Get parameters
        batch_size = int(request.form.get('batch_size', 10))
        threads = int(request.form.get('threads', 4))
        size_x = int(request.form.get('size_x', 20))
        size_y = int(request.form.get('size_y', 20))
        size_z = int(request.form.get('size_z', 20))
        thread_pocket = int(request.form.get('thread_pocket', 1000))
        model_type = request.form.get('model_type', 'alphafold')
        
        # Create temporary directory for results
        output_dir = tempfile.mkdtemp()
        logger.info(f"Created output directory at {output_dir}")
        
        # Create protein_files.txt
        protein_files_txt = create_protein_files_txt(output_dir)
        if not protein_files_txt:
            return jsonify({'error': 'Error creating protein files list'}), 500
        
        # Save ligands file
        ligands_path = save_uploaded_file(ligands_file)
        
        # Validate ligands file format
        try:
            with open(ligands_path, 'r') as f:
                first_line = f.readline().strip()
                if ',' not in first_line:
                    return jsonify({'error': 'Ligands file must be in CSV format with "molecule_name,smiles" header'}), 400
        except Exception as e:
            return jsonify({'error': f'Error reading ligands file: {str(e)}'}), 400
        
        # Configure PocketVina
        config = PocketVinaConfig(
            protein_paths=protein_files_txt,
            smiles_file=ligands_path,
            output_dir=output_dir,
            batch_size=batch_size,
            p2rank_config=P2RankConfig(         
                threads=threads,
                visualizations=True,
                vis_copy_proteins=True,
                output_dir=output_dir,
                model=model_type
            ),
            pocket_config=PocketConfig(
                size_x=size_x,
                size_y=size_y,
                size_z=size_z,
                thread=thread_pocket
            )
        )
        
        # Initialize and run PocketVina
        logger.info("Starting PocketVina docking...")
        pocketvina = PocketVinaGPU(config, htvs_mode=True)
        pocketvina.execute_docking()
        logger.info("Docking completed")
        
        # Scan for visualization files created during docking
        scan_visualization_files(output_dir)
        
        # Read and return results
        results_path = os.path.join(output_dir, "combined_results.csv")
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            
            # Debug: Log the directory structure and CSV content
            logger.info(f"🗂️ Output directory structure:")
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                logger.info(f"{indent}{os.path.basename(root)}/")
                sub_indent = ' ' * 2 * (level + 1)
                for file in files[:10]:  # Limit to first 10 files per directory
                    logger.info(f"{sub_indent}{file}")
                if len(files) > 10:
                    logger.info(f"{sub_indent}... and {len(files) - 10} more files")
            
            # Debug: Log CSV columns and sample data
            logger.info(f"📊 CSV columns: {list(df.columns)}")
            if not df.empty:
                logger.info(f"📄 First row sample:")
                for col in df.columns:
                    logger.info(f"  {col}: {df.iloc[0][col]}")
            
            # Fix paths for new PocketVina directory structure
            logger.info("🔧 Fixing result paths for new PocketVina structure...")
            for index, row in df.iterrows():
                for col in df.columns:
                    if col.lower() in ['ligand', 'ligand_file', 'output_file', 'pdbqt_file', 'result_file']:
                        current_path = str(row[col])
                        if current_path and current_path != 'nan' and current_path != 'N/A':
                            # Try to find the actual file in batch_*_out directories
                            filename = os.path.basename(current_path)
                            logger.info(f"🔍 Looking for PDBQT file: {filename}")
                            
                            # Search in batch_*_out directories
                            found_path = None
                            for item in os.listdir(output_dir):
                                item_path = os.path.join(output_dir, item)
                                if os.path.isdir(item_path) and item.startswith('batch_') and item.endswith('_out'):
                                    batch_dir = item_path
                                    for root, dirs, files in os.walk(batch_dir):
                                        if filename in files:
                                            found_path = os.path.join(root, filename)
                                            logger.info(f"✅ Found PDBQT at: {found_path}")
                                            break
                                        # Also try partial matching
                                        for file in files:
                                            if filename.replace('.pdbqt', '') in file and file.endswith('.pdbqt'):
                                                found_path = os.path.join(root, file)
                                                logger.info(f"✅ Found similar PDBQT at: {found_path}")
                                                break
                                    if found_path:
                                        break
                            
                            if found_path:
                                df.at[index, col] = found_path
                                logger.info(f"🔗 Fixed path: {current_path} -> {found_path}")
                            else:
                                logger.warning(f"⚠️ Could not find PDBQT file for: {current_path}")
            
            # Build best-pose CSV (unique protein/molecule pair with lowest affinity)
            def pick_column(columns, exact_candidates, keyword_candidates):
                lower_map = {c.lower(): c for c in columns}
                # exact lower-case matches
                for cand in exact_candidates:
                    if cand in lower_map:
                        return lower_map[cand]
                # fuzzy keyword contains
                for col in columns:
                    lc = col.lower()
                    for kw in keyword_candidates:
                        if kw in lc:
                            return col
                return None

            protein_col = pick_column(
                df.columns,
                exact_candidates=['protein', 'protein_file', 'protein_path'],
                keyword_candidates=['protein', 'prot']
            )
            ligand_name_col = pick_column(
                df.columns,
                exact_candidates=['molecule_name', 'ligand_name', 'ligand', 'ligand_file'],
                keyword_candidates=['molecule_name', 'molecule', 'ligand']
            )
            score_col = pick_column(
                df.columns,
                exact_candidates=['affinity (kcal/mol)', 'affinity', 'vina_score', 'binding_affinity', 'score'],
                keyword_candidates=['affinity', 'kcal', 'score']
            )

            best_csv_path = os.path.join(output_dir, "best_pose_results.csv")
            df_best = df
            try:
                if protein_col and ligand_name_col and score_col:
                    df_numeric = df.copy()
                    df_numeric[score_col] = pd.to_numeric(df_numeric[score_col], errors='coerce')
                    df_numeric = df_numeric.dropna(subset=[score_col])
                    # pick lowest (most negative) score per protein/ligand
                    idx = df_numeric.groupby([protein_col, ligand_name_col])[score_col].idxmin()
                    df_best = df_numeric.loc[idx].sort_values(by=[protein_col, ligand_name_col]).reset_index(drop=True)
                df_best.to_csv(best_csv_path, index=False)
            except Exception as e:
                logger.warning(f"Could not compute best-pose CSV: {e}")
                df.to_csv(best_csv_path, index=False)

            # Convert DataFrame to sortable HTML table (display best only)
            results_html = create_sortable_table(df_best, output_dir=output_dir)
            
            success_msg = f"Docking completed successfully! Processed {len(uploaded_proteins)} proteins."
            logger.info(success_msg)
            
            return jsonify({
                'success': True,
                'message': success_msg,
                'results_html': results_html,
                'results_path_all': results_path,
                'results_path_best': best_csv_path,
                'output_dir': output_dir  # Send output dir for file access
            })
        else:
            return jsonify({'error': 'Docking completed but no results file found'}), 500
            
    except Exception as e:
        error_msg = f"Error occurred: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500

@app.route('/download_results')
def download_results():
    results_path = request.args.get('path')
    download_name = request.args.get('name', 'results.csv')
    if results_path and os.path.exists(results_path):
        return send_file(results_path, as_attachment=True, download_name=download_name)
    return jsonify({'error': 'Results file not found'}), 404

@app.route('/view_protein/<path:filename>')
def view_protein(filename):
    """Serve protein files for 3D visualization"""
    try:
        # Security check - ensure file exists in uploaded proteins
        global uploaded_proteins
        file_path = None
        
        logger.info(f"Looking for protein file: {filename}")
        logger.info(f"Available proteins: {[p['filename'] for p in uploaded_proteins]}")
        
        # Try exact match first
        for protein_info in uploaded_proteins:
            if protein_info['filename'] == filename:
                file_path = protein_info['file_path']
                logger.info(f"Found exact matching protein: {file_path}")
                break
        
        # If no exact match, try fuzzy matching (remove extensions and suffixes)
        if not file_path:
            logger.info(f"No exact match for '{filename}', trying fuzzy matching...")
            search_name = filename.replace('.pdb', '').replace('-pocket1', '').replace('-pocket2', '').replace('-pocket3', '')
            
            for protein_info in uploaded_proteins:
                original_name = protein_info['original_name'].replace('.pdb', '')
                stored_name = protein_info['filename'].replace('.pdb', '')
                
                if (search_name in stored_name or 
                    original_name in search_name or 
                    stored_name in search_name):
                    file_path = protein_info['file_path']
                    logger.info(f"Found fuzzy matching protein: '{filename}' -> '{file_path}'")
                    break
        
        if not file_path:
            logger.error(f"Protein file {filename} not found in uploaded proteins list")
            logger.error(f"Available proteins: {[p['filename'] for p in uploaded_proteins]}")
            return jsonify({'error': f'Protein file {filename} not found'}), 404
            
        if not os.path.exists(file_path):
            logger.error(f"Protein file path {file_path} does not exist on filesystem")
            return jsonify({'error': f'Protein file {filename} not found on disk'}), 404
            
        logger.info(f"Serving protein file: {file_path}")
        return send_file(file_path, as_attachment=False, mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Error serving protein file {filename}: {e}")
        return jsonify({'error': f'Error loading protein file: {str(e)}'}), 500

@app.route('/view_result_file')
def view_result_file():
    """Serve result files (PDBQT) for 3D visualization"""
    try:
        file_path = request.args.get('path')
        logger.info(f"🔍 Serving result file: {file_path}")
        
        if not file_path:
            return jsonify({'error': 'No file path provided'}), 400
            
        if not os.path.exists(file_path):
            logger.error(f"❌ Result file not found: {file_path}")
            # List files in parent directory for debugging
            parent_dir = os.path.dirname(file_path)
            if os.path.exists(parent_dir):
                available_files = os.listdir(parent_dir)
                logger.info(f"📁 Available files in {parent_dir}: {available_files}")
            return jsonify({'error': f'Result file not found: {file_path}'}), 404
        
        # Check file size and content for PDBQT files
        file_size = os.path.getsize(file_path)
        is_pdbqt = file_path.lower().endswith('.pdbqt')
        
        logger.info(f"📊 File stats - Size: {file_size} bytes, PDBQT: {is_pdbqt}")
        
        if file_size == 0:
            logger.error(f"❌ Result file is empty: {file_path}")
            return jsonify({'error': 'Result file is empty'}), 404
        
        if is_pdbqt and file_size < 100:
            logger.warning(f"⚠️ PDBQT file is very small ({file_size} bytes), may not contain ligand data")
        
        # Log first few lines for debugging
        try:
            with open(file_path, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(3)]
                logger.info(f"📄 First lines of file: {first_lines}")
        except Exception as read_error:
            logger.warning(f"Could not read file content for debugging: {read_error}")
            
        logger.info(f"✅ Successfully serving result file: {file_path}")
        return send_file(file_path, as_attachment=False, mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"❌ Error serving result file {file_path}: {e}")
        return jsonify({'error': f'Error loading result file: {str(e)}'}), 500

@app.route('/view_processed_file')
def view_processed_file():
    """Serve processed visualization files (PDB, scripts)"""
    try:
        file_path = request.args.get('path')
        logger.info(f"Serving processed file: {file_path}")
        
        if not file_path:
            return jsonify({'error': 'No file path provided'}), 400
            
        if not os.path.exists(file_path):
            logger.error(f"Processed file not found: {file_path}")
            return jsonify({'error': f'Processed file not found: {file_path}'}), 404
        
        # Determine mimetype based on extension
        if file_path.endswith('.pdb'):
            mimetype = 'text/plain'
        elif file_path.endswith('.cxc'):
            mimetype = 'text/plain'  # ChimeraX script
        elif file_path.endswith('.pml'):
            mimetype = 'text/plain'  # PyMOL script
        else:
            mimetype = 'text/plain'
            
        logger.info(f"Successfully serving processed file: {file_path}")
        return send_file(file_path, as_attachment=False, mimetype=mimetype)
        
    except Exception as e:
        logger.error(f"Error serving processed file {file_path}: {e}")
        return jsonify({'error': f'Error loading processed file: {str(e)}'}), 500

@app.route('/view_script')
def view_script():
    """View script content for debugging"""
    try:
        script_path = request.args.get('path')
        script_type = request.args.get('type', 'chimerax')
        
        if not script_path or not os.path.exists(script_path):
            return jsonify({'error': 'Script file not found'}), 404
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        return jsonify({
            'success': True,
            'script_type': script_type,
            'content': content,
            'path': script_path
        })
        
    except Exception as e:
        logger.error(f"Error reading script {script_path}: {e}")
        return jsonify({'error': f'Error reading script: {str(e)}'}), 500

if __name__ == '__main__':
    # Get port from environment for Hugging Face compatibility
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False) 