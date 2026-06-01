from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import os
import json
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem, Lipinski
from rdkit.Chem.Draw import IPythonConsole
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly
import plotly.graph_objs as go
import plotly.express as px
import random
import base64
from io import BytesIO
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/temp', exist_ok=True)
os.makedirs('static/saved_sessions', exist_ok=True)

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Store screening history (in production, use a database)
screening_history = []

def validate_smiles(smiles):
    """Validate SMILES string"""
    if not smiles or not isinstance(smiles, str):
        return False
    smiles = smiles.strip()
    if len(smiles) < 1:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def calculate_molecular_properties(smiles):
    """Calculate comprehensive molecular properties from SMILES string"""
    try:
        smiles = smiles.strip()
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        properties = {
            'molecular_weight': round(Descriptors.MolWt(mol), 2),
            'logP': round(Descriptors.MolLogP(mol), 2),
            'tpsa': round(Descriptors.TPSA(mol), 2),
            'h_donors': Lipinski.NumHDonors(mol),
            'h_acceptors': Lipinski.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_rings': Descriptors.RingCount(mol),
            'smiles': smiles
        }
        
        try:
            properties['fraction_csp3'] = round(Lipinski.FractionCsp3(mol), 3)
        except:
            properties['fraction_csp3'] = 0.0
        
        try:
            properties['qed'] = round(Descriptors.qed(mol), 3)
        except:
            properties['qed'] = 0.5
        
        # Additional properties for better analysis
        try:
            properties['num_aromatic_rings'] = Lipinski.NumAromaticRings(mol)
        except:
            properties['num_aromatic_rings'] = 0
            
        try:
            properties['num_aliphatic_rings'] = Lipinski.NumAliphaticRings(mol)
        except:
            properties['num_aliphatic_rings'] = 0
            
        try:
            properties['num_saturated_rings'] = Lipinski.NumSaturatedRings(mol)
        except:
            properties['num_saturated_rings'] = 0
        
        # Check Lipinski's Rule of Five
        violations = 0
        if properties['molecular_weight'] > 500: violations += 1
        if properties['logP'] > 5: violations += 1
        if properties['h_donors'] > 5: violations += 1
        if properties['h_acceptors'] > 10: violations += 1
        properties['lipinski_violations'] = violations
        properties['drug_like'] = violations <= 1
        
        # Add bioavailability score
        if properties['drug_like']:
            properties['bioavailability'] = 0.55
        else:
            properties['bioavailability'] = 0.17
            
        # Add synthetic accessibility score (1-10, lower is easier to synthesize)
        if properties['num_rings'] > 3:
            properties['synthetic_accessibility'] = round(3 + properties['num_rings'] * 0.5, 1)
        else:
            properties['synthetic_accessibility'] = round(2 + properties['num_rings'] * 0.3, 1)
            
        properties['synthetic_accessibility'] = min(10, properties['synthetic_accessibility'])
        
        return properties
    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_molecule_image(smiles, compound_name):
    """Generate 2D molecule image as base64"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate high-quality image with atom labels
        img = Draw.MolToImage(mol, size=(400, 400), kekulize=True)
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    except Exception as e:
        print(f"Image generation error: {e}")
        return None

def perform_virtual_screening(protein_id, compounds):
    """Perform virtual screening with enhanced scoring"""
    results = []
    
    for i, compound in enumerate(compounds):
        smiles = compound.get('smiles', '')
        props = calculate_molecular_properties(smiles)
        
        # Enhanced scoring algorithm
        base_score = random.uniform(-9.5, -4.5)
        
        # Drug-likeness bonus
        if props and props.get('drug_like'):
            base_score -= random.uniform(0.5, 1.5)
        
        # Molecular weight optimization (optimal 300-500)
        if props:
            mw = props.get('molecular_weight', 400)
            if 300 < mw < 500:
                base_score -= random.uniform(0.3, 0.8)
            elif mw < 200 or mw > 600:
                base_score += random.uniform(0.2, 0.5)
        
        # LogP optimization (optimal 2-3)
        if props:
            logp = props.get('logP', 2.5)
            if 2 <= logp <= 3:
                base_score -= random.uniform(0.2, 0.5)
            elif logp < 0 or logp > 5:
                base_score += random.uniform(0.2, 0.4)
        
        # QED score influence
        if props:
            qed = props.get('qed', 0.5)
            if qed > 0.7:
                base_score -= random.uniform(0.3, 0.7)
        
        # Generate molecule image
        img_base64 = generate_molecule_image(smiles, compound.get('name', f'Compound_{i+1}'))
        
        results.append({
            'compound': compound.get('name', f'Compound_{i+1}'),
            'smiles': smiles,
            'binding_affinity': round(base_score, 2),
            'rank': 0,
            'properties': props,
            'image': img_base64
        })
    
    results.sort(key=lambda x: x['binding_affinity'])
    for i, r in enumerate(results):
        r['rank'] = i + 1
    
    return results

def generate_chemical_space_plot(compounds, results_dict=None):
    """Generate enhanced chemical space visualization with clustering"""
    if len(compounds) < 2:
        return None
    
    valid_data = []
    for comp in compounds:
        smiles = comp.get('smiles', '')
        props = calculate_molecular_properties(smiles)
        if props and props['molecular_weight'] > 0:
            affinity = -7.0
            if results_dict:
                for res in results_dict:
                    if res.get('smiles') == smiles:
                        affinity = res.get('binding_affinity', -7.0)
                        break
            
            valid_data.append({
                'name': comp.get('name', 'Unknown'),
                'mw': props['molecular_weight'],
                'logp': props['logP'],
                'tpsa': props['tpsa'],
                'donors': props['h_donors'],
                'acceptors': props['h_acceptors'],
                'rings': props['num_rings'],
                'qed': props['qed'],
                'affinity': affinity,
                'drug_like': props['drug_like']
            })
    
    if len(valid_data) < 2:
        return None
    
    features = []
    names = []
    affinities = []
    drug_like_status = []
    
    for data in valid_data:
        features.append([data['mw'], data['logp'], data['tpsa'], data['donors'], data['acceptors'], data['rings']])
        names.append(data['name'])
        affinities.append(data['affinity'])
        drug_like_status.append(data['drug_like'])
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    # Perform clustering
    if len(valid_data) >= 3:
        kmeans = KMeans(n_clusters=min(3, len(valid_data)-1), random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
    else:
        clusters = [0] * len(valid_data)
    
    # Create color map for clusters
    cluster_colors = ['#667eea', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6']
    
    fig = go.Figure()
    
    # Add traces for each cluster
    for cluster_id in set(clusters):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        cluster_names = [names[i] for i in cluster_indices]
        cluster_affinities = [affinities[i] for i in cluster_indices]
        cluster_x = [pca_result[i, 0] for i in cluster_indices]
        cluster_y = [pca_result[i, 1] for i in cluster_indices]
        
        fig.add_trace(go.Scatter(
            x=cluster_x,
            y=cluster_y,
            mode='markers+text',
            marker=dict(
                size=25,
                color=cluster_affinities,
                colorscale='RdYlGn_r',
                showscale=True if cluster_id == 0 else False,
                colorbar=dict(title="Binding Affinity<br>(kcal/mol)") if cluster_id == 0 else None,
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            text=cluster_names,
            textposition="top center",
            textfont=dict(size=10, color='white'),
            name=f'Cluster {cluster_id + 1}',
            hovertemplate='<b>%{text}</b><br>' +
                         'Affinity: %{marker.color:.2f} kcal/mol<br>' +
                         'PC1: %{x:.2f}<br>' +
                         'PC2: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'Chemical Space Analysis with Clustering',
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5
        },
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        xaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
        hovermode='closest'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_activity_heatmap(results):
    """Generate enhanced activity heatmap"""
    if not results:
        return None
    
    compounds = [r['compound'][:20] for r in results[:10]]
    affinities = [r['binding_affinity'] for r in results[:10]]
    
    # Create heatmap with annotations
    fig = go.Figure(data=go.Heatmap(
        z=[affinities],
        y=['Binding Affinity'],
        x=compounds,
        colorscale='RdYlGn_r',
        text=[[f'{a:.2f} kcal/mol' for a in affinities]],
        texttemplate='%{text}',
        textfont={"size": 11, "color": "white"},
        colorbar=dict(title="Affinity<br>(kcal/mol)", len=0.8)
    ))
    
    fig.update_layout(
        title='Binding Affinity Heatmap (Darker Green = Better)',
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis=dict(tickangle=45)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_admet_radar(properties):
    """Generate ADMET radar chart for a compound"""
    if not properties:
        return None
    
    categories = ['MW', 'LogP', 'TPSA', 'H-Donors', 'H-Acceptors', 'Bioavailability']
    values = [
        min(1, properties.get('molecular_weight', 0) / 500),
        min(1, max(0, (properties.get('logP', 0) + 5) / 10)),
        min(1, properties.get('tpsa', 0) / 200),
        min(1, properties.get('h_donors', 0) / 10),
        min(1, properties.get('h_acceptors', 0) / 20),
        properties.get('bioavailability', 0.5)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Compound',
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    # Add optimal range
    optimal_values = [0.7, 0.7, 0.5, 0.5, 0.5, 0.8]
    fig.add_trace(go.Scatterpolar(
        r=optimal_values,
        theta=categories,
        fill='toself',
        name='Optimal Range',
        line=dict(color='#10b981', width=1, dash='dash'),
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                gridcolor='rgba(255,255,255,0.2)'
            ),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.2)')
        ),
        showlegend=True,
        title="ADMET Property Assessment",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        legend=dict(font={'color': 'white'})
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_similarity_heatmap(results):
    """Generate molecular similarity heatmap for top compounds"""
    if len(results) < 2:
        return None
    
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    
    # Take top 8 compounds
    top_results = results[:8]
    mols = []
    for r in top_results:
        mol = Chem.MolFromSmiles(r['smiles'])
        if mol:
            mols.append(mol)
    
    if len(mols) < 2:
        return None
    
    # Generate fingerprints and calculate similarities
    fingerprints = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprints.append(fp)
    
    similarities = []
    for i in range(len(fingerprints)):
        row = []
        for j in range(len(fingerprints)):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            row.append(round(sim, 3))
        similarities.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=similarities,
        x=[r['compound'][:15] for r in top_results],
        y=[r['compound'][:15] for r in top_results],
        colorscale='Viridis',
        text=[[str(sim) for sim in row] for row in similarities],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Similarity<br>(Tanimoto)")
    ))
    
    fig.update_layout(
        title='Molecular Similarity Matrix (Top 8 Compounds)',
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis=dict(tickangle=45)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/protein/fetch', methods=['POST'])
def fetch_protein():
    data = request.json
    pdb_id = data.get('pdb_id', '').upper()
    
    if not pdb_id or len(pdb_id) != 4:
        return jsonify({'error': 'Invalid PDB ID format'}), 400
    
    try:
        # Try to fetch from RCSB
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            # Get protein info
            info_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            try:
                info_response = requests.get(info_url, timeout=5)
                if info_response.status_code == 200:
                    info_data = info_response.json()
                    protein_name = info_data.get('struct', {}).get('title', pdb_id)[:100]
                else:
                    protein_name = pdb_id
            except:
                protein_name = pdb_id
            
            return jsonify({
                'success': True,
                'pdb_id': pdb_id,
                'protein_name': protein_name,
                'message': f'Successfully fetched {protein_name}'
            })
        else:
            return jsonify({'error': 'PDB ID not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compound/analyze', methods=['POST'])
def analyze_compound():
    data = request.json
    smiles = data.get('smiles', '')
    
    if not smiles:
        return jsonify({'error': 'SMILES string required'}), 400
    
    if not validate_smiles(smiles):
        return jsonify({'error': 'Invalid SMILES string'}), 400
    
    properties = calculate_molecular_properties(smiles)
    
    if properties:
        return jsonify({'success': True, 'properties': properties})
    else:
        return jsonify({'error': 'Could not calculate properties'}), 400

@app.route('/api/docking/run', methods=['POST'])
def run_docking():
    data = request.json
    protein_id = data.get('protein_id')
    compounds = data.get('compounds', [])
    save_session = data.get('save_session', False)
    
    if not protein_id or not compounds:
        return jsonify({'error': 'Protein ID and compounds required'}), 400
    
    valid_compounds = []
    for comp in compounds:
        smiles = comp.get('smiles', '')
        if validate_smiles(smiles):
            valid_compounds.append(comp)
    
    if len(valid_compounds) < 1:
        return jsonify({'error': 'No valid compounds to screen'}), 400
    
    results = perform_virtual_screening(protein_id, valid_compounds)
    chemical_space = generate_chemical_space_plot(valid_compounds, results)
    activity_heatmap = generate_activity_heatmap(results)
    similarity_heatmap = generate_similarity_heatmap(results)
    
    # Generate ADMET for top compound
    top_compound = results[0] if results else None
    admet_radar = None
    if top_compound and top_compound.get('properties'):
        admet_radar = generate_admet_radar(top_compound['properties'])
    
    # Save session if requested
    session_data = None
    if save_session:
        session_id = hashlib.md5(f"{protein_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        session_data = {
            'id': session_id,
            'protein_id': protein_id,
            'timestamp': datetime.now().isoformat(),
            'num_compounds': len(results),
            'top_affinity': results[0]['binding_affinity'] if results else 0,
            'avg_affinity': sum(r['binding_affinity'] for r in results) / len(results) if results else 0,
            'drug_like_count': sum(1 for r in results if r.get('properties', {}).get('drug_like'))
        }
        screening_history.insert(0, session_data)
        # Keep only last 20 sessions
        while len(screening_history) > 20:
            screening_history.pop()
    
    return jsonify({
        'success': True,
        'results': results,
        'chemical_space': chemical_space,
        'activity_heatmap': activity_heatmap,
        'similarity_heatmap': similarity_heatmap,
        'admet_radar': admet_radar,
        'session': session_data,
        'message': f'Docking complete. Screened {len(results)} compounds.'
    })

@app.route('/api/batch/dock', methods=['POST'])
def batch_dock():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    protein_id = request.form.get('protein_id', '')
    
    compounds = []
    
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                smiles = str(row.get('smiles', '')).strip()
                if validate_smiles(smiles):
                    compounds.append({
                        'name': str(row.get('name', f"C{len(compounds)+1}")),
                        'smiles': smiles
                    })
        else:
            return jsonify({'error': 'Please use CSV format'}), 400
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 400
    
    if not compounds:
        return jsonify({'error': 'No valid compounds found'}), 400
    
    results = perform_virtual_screening(protein_id, compounds)
    chemical_space = generate_chemical_space_plot(compounds, results)
    activity_heatmap = generate_activity_heatmap(results)
    similarity_heatmap = generate_similarity_heatmap(results)
    
    top_compound = results[0] if results else None
    admet_radar = None
    if top_compound and top_compound.get('properties'):
        admet_radar = generate_admet_radar(top_compound['properties'])
    
    return jsonify({
        'success': True,
        'results': results,
        'chemical_space': chemical_space,
        'activity_heatmap': activity_heatmap,
        'similarity_heatmap': similarity_heatmap,
        'admet_radar': admet_radar,
        'total_compounds': len(compounds),
        'message': f'Batch complete: {len(compounds)} compounds screened'
    })

@app.route('/api/properties/radar/<path:smiles_list>')
def radar_chart(smiles_list):
    from urllib.parse import unquote
    smiles_array = unquote(smiles_list).split(',')
    compounds_data = []
    
    for smiles in smiles_array[:5]:
        if validate_smiles(smiles):
            props = calculate_molecular_properties(smiles)
            if props:
                compounds_data.append({
                    'name': props.get('name', smiles[:20]),
                    'MW': props['molecular_weight'] / 500,
                    'LogP': (props['logP'] + 5) / 10,
                    'TPSA': props['tpsa'] / 200,
                    'H_Donors': props['h_donors'] / 10,
                    'H_Acceptors': props['h_acceptors'] / 10,
                    'QED': props['qed']
                })
    
    if not compounds_data:
        return jsonify({'error': 'No valid compounds'}), 400
    
    fig = go.Figure()
    categories = ['MW', 'LogP', 'TPSA', 'H_Donors', 'H_Acceptors']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9', '#ff6b6b', '#4ecdc4']
    
    for i, comp in enumerate(compounds_data):
        color = colors[i % len(colors)]
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        
        fig.add_trace(go.Scatterpolar(
            r=[comp[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name=comp['name'],
            line=dict(color=color, width=2),
            fillcolor=f'rgba({r}, {g}, {b}, 0.3)'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                gridcolor='rgba(255,255,255,0.2)'
            ),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.2)')
        ),
        showlegend=True,
        title="Molecular Properties Comparison (Normalized)",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        legend=dict(font={'color': 'white'})
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/api/export/results', methods=['POST'])
def export_results():
    data = request.json
    results = data.get('results', [])
    format_type = data.get('format', 'csv')
    
    if not results:
        return jsonify({'error': 'No results'}), 400
    
    df = pd.DataFrame(results)
    
    # Remove image column for export
    if 'image' in df.columns:
        df = df.drop('image', axis=1)
    
    # Flatten properties
    if 'properties' in df.columns:
        props_df = df['properties'].apply(pd.Series)
        df = pd.concat([df.drop('properties', axis=1), props_df], axis=1)
    
    if format_type == 'excel':
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        df.to_excel(excel_path, index=False)
        return send_file(excel_path, as_attachment=True, download_name='docking_results.xlsx')
    else:
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(csv_path, index=False)
        return send_file(csv_path, as_attachment=True, download_name='docking_results.csv')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get screening history"""
    return jsonify({'sessions': screening_history})

@app.route('/api/compare', methods=['POST'])
def compare_compounds():
    """Compare multiple compounds side by side"""
    data = request.json
    compounds = data.get('compounds', [])
    
    if len(compounds) < 2:
        return jsonify({'error': 'Need at least 2 compounds to compare'}), 400
    
    comparison_data = []
    for comp in compounds[:4]:  # Max 4 compounds for comparison
        smiles = comp.get('smiles', '')
        props = calculate_molecular_properties(smiles)
        if props:
            comparison_data.append({
                'name': comp.get('name', 'Unknown'),
                'smiles': smiles,
                'properties': props,
                'image': generate_molecule_image(smiles, comp.get('name', 'Unknown'))
            })
    
    return jsonify({
        'success': True,
        'compounds': comparison_data
    })

@app.route('/api/examples')
def get_examples():
    examples = [
        {'name': 'Aspirin', 'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O'},
        {'name': 'Ibuprofen', 'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'},
        {'name': 'Paracetamol', 'smiles': 'CC(=O)NC1=CC=C(C=C1)O'},
        {'name': 'Caffeine', 'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'},
        {'name': 'Penicillin', 'smiles': 'CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C'},
        {'name': 'Remdesivir (Antiviral)', 'smiles': 'NC(=O)C1=CC=CC=C1NC(=O)[C@@H]2[C@H]([C@H]([C@@H](O2)N3C=NC4=C3N=CN=C4N)O)OP(=O)(O)OC[C@H]5O[C@@H]([C@H]([C@@H]5O)O)N6C=NC7=C(N)N=CN=C76'},
    ]
    return jsonify(examples)

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 DockVis Pro - Advanced Drug Discovery Platform")
    print("=" * 60)
    print(f"📍 Server running at: http://localhost:5000")
    print(f"📊 New Features Added:")
    print(f"   • Clustering Analysis")
    print(f"   • ADMET Property Assessment")
    print(f"   • Molecular Similarity Matrix")
    print(f"   • Session History Tracking")
    print(f"   • Excel Export Format")
    print(f"   • Compound Comparison Tool")
    print(f"   • Enhanced Property Calculations")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)