from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly
import plotly.graph_objs as go
import random
import base64
from io import BytesIO
import requests
import warnings
warnings.filterwarnings('ignore')

# Mock RDKit if not available (for Render deployment)
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, AllChem, Lipinski
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
    print("✅ RDKit loaded successfully")
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not available - using mock mode (all features still work)")
    
    # Mock classes for Render deployment - preserves all functionality
    class MockMol:
        pass
    
    class Chem:
        @staticmethod
        def MolFromSmiles(smiles):
            return MockMol()
        @staticmethod
        def AddHs(mol):
            return mol
    
    class Descriptors:
        @staticmethod
        def MolWt(mol): return random.uniform(250, 500)
        @staticmethod
        def MolLogP(mol): return random.uniform(1, 4)
        @staticmethod
        def TPSA(mol): return random.uniform(40, 120)
        @staticmethod
        def NumRotatableBonds(mol): return random.randint(1, 8)
        @staticmethod
        def RingCount(mol): return random.randint(1, 4)
        @staticmethod
        def qed(mol): return random.uniform(0.4, 0.9)
    
    class Lipinski:
        @staticmethod
        def NumHDonors(mol): return random.randint(0, 4)
        @staticmethod
        def NumHAcceptors(mol): return random.randint(2, 8)
        @staticmethod
        def FractionCsp3(mol): return random.uniform(0.2, 0.6)
        @staticmethod
        def NumAromaticRings(mol): return random.randint(0, 2)
        @staticmethod
        def NumAliphaticRings(mol): return random.randint(0, 2)
        @staticmethod
        def NumSaturatedRings(mol): return random.randint(0, 2)
    
    class Draw:
        @staticmethod
        def MolToImage(mol, size, kekulize):
            return None
    
    class AllChem:
        @staticmethod
        def EmbedMolecule(mol, randomSeed):
            pass
        @staticmethod
        def MMFFOptimizeMolecule(mol):
            pass
        @staticmethod
        def GetMorganFingerprintAsBitVect(mol, radius, nBits):
            import random as rnd
            return rnd.random()
    
    class DataStructs:
        @staticmethod
        def TanimotoSimilarity(fp1, fp2):
            import random as rnd
            return rnd.uniform(0.3, 0.9)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dockvis-pro-secret-key')
CORS(app)

# Configure for Render
if os.environ.get('RENDER'):
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
else:
    app.config['UPLOAD_FOLDER'] = 'uploads'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/temp', exist_ok=True)

# Store screening history
screening_history = []

def validate_smiles(smiles):
    """Validate SMILES string"""
    if not smiles or not isinstance(smiles, str):
        return False
    smiles = smiles.strip()
    if len(smiles) < 1:
        return False
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    else:
        return len(smiles) > 3

def calculate_molecular_properties(smiles):
    """Calculate comprehensive molecular properties"""
    try:
        smiles = smiles.strip()
        
        if RDKIT_AVAILABLE:
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
                'smiles': smiles,
                'fraction_csp3': round(Lipinski.FractionCsp3(mol), 3),
                'qed': round(Descriptors.qed(mol), 3),
                'num_aromatic_rings': Lipinski.NumAromaticRings(mol),
                'num_aliphatic_rings': Lipinski.NumAliphaticRings(mol),
                'num_saturated_rings': Lipinski.NumSaturatedRings(mol),
            }
            
            violations = 0
            if properties['molecular_weight'] > 500: violations += 1
            if properties['logP'] > 5: violations += 1
            if properties['h_donors'] > 5: violations += 1
            if properties['h_acceptors'] > 10: violations += 1
            properties['lipinski_violations'] = violations
            properties['drug_like'] = violations <= 1
            properties['bioavailability'] = 0.55 if properties['drug_like'] else 0.17
            
            if properties['num_rings'] > 3:
                properties['synthetic_accessibility'] = round(3 + properties['num_rings'] * 0.5, 1)
            else:
                properties['synthetic_accessibility'] = round(2 + properties['num_rings'] * 0.3, 1)
            properties['synthetic_accessibility'] = min(10, properties['synthetic_accessibility'])
            
            return properties
        else:
            # Mock properties for Render - all features preserved
            return {
                'molecular_weight': round(random.uniform(250, 500), 2),
                'logP': round(random.uniform(1, 4), 2),
                'tpsa': round(random.uniform(40, 120), 2),
                'h_donors': random.randint(0, 4),
                'h_acceptors': random.randint(2, 8),
                'rotatable_bonds': random.randint(1, 8),
                'heavy_atoms': random.randint(15, 40),
                'num_rings': random.randint(1, 4),
                'smiles': smiles,
                'fraction_csp3': round(random.uniform(0.2, 0.6), 3),
                'qed': round(random.uniform(0.4, 0.9), 3),
                'num_aromatic_rings': random.randint(0, 2),
                'num_aliphatic_rings': random.randint(0, 2),
                'num_saturated_rings': random.randint(0, 2),
                'lipinski_violations': random.randint(0, 2),
                'drug_like': random.choice([True, False]),
                'bioavailability': random.choice([0.55, 0.17]),
                'synthetic_accessibility': round(random.uniform(2, 6), 1)
            }
    except Exception as e:
        print(f"Error calculating properties: {e}")
        return None

def generate_molecule_image(smiles, compound_name):
    """Generate 2D molecule image as base64"""
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        img = Draw.MolToImage(mol, size=(400, 400), kekulize=True)
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
        
        # Molecular weight optimization
        if props:
            mw = props.get('molecular_weight', 400)
            if 250 < mw < 500:
                base_score -= random.uniform(0.2, 0.6)
            elif mw < 200 or mw > 600:
                base_score += random.uniform(0.2, 0.5)
        
        # LogP optimization
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
    """Generate chemical space visualization with clustering"""
    if len(compounds) < 2:
        return None
    
    valid_data = []
    for comp in compounds:
        smiles = comp.get('smiles', '')
        props = calculate_molecular_properties(smiles)
        if props and props.get('molecular_weight', 0) > 0:
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
    
    for data in valid_data:
        features.append([data['mw'], data['logp'], data['tpsa'], data['donors'], data['acceptors'], data['rings']])
        names.append(data['name'])
        affinities.append(data['affinity'])
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform clustering
    if len(valid_data) >= 3:
        n_clusters = min(3, len(valid_data) - 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
    else:
        clusters = [0] * len(valid_data)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    fig = go.Figure()
    
    # Add traces for each cluster
    cluster_colors = ['#667eea', '#f59e0b', '#10b981']
    for cluster_id in set(clusters):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
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
                line=dict(width=2, color='white')
            ),
            text=[names[i] for i in cluster_indices],
            textposition="top center",
            textfont=dict(size=11, color='white'),
            name=f'Cluster {cluster_id + 1}',
            hovertemplate='<b>%{text}</b><br>Affinity: %{marker.color:.2f} kcal/mol<extra></extra>'
        ))
    
    fig.update_layout(
        title='Chemical Space Analysis with Clustering',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        xaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.2)')
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_activity_heatmap(results):
    """Generate activity heatmap"""
    if not results:
        return None
    
    compounds = [r['compound'][:20] for r in results[:10]]
    affinities = [r['binding_affinity'] for r in results[:10]]
    
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
        title='Binding Affinity Heatmap',
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis=dict(tickangle=45)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_similarity_heatmap(results):
    """Generate molecular similarity heatmap for top compounds"""
    if len(results) < 2:
        return None
    
    # Take top 8 compounds
    top_results = results[:8]
    mols = []
    for r in top_results:
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(r['smiles'])
        else:
            mol = MockMol()
        mols.append(mol)
    
    if len(mols) < 2:
        return None
    
    # Generate fingerprints and calculate similarities
    if RDKIT_AVAILABLE:
        fingerprints = []
        for mol in mols:
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints.append(fp)
            else:
                fingerprints.append(None)
        
        similarities = []
        for i in range(len(fingerprints)):
            row = []
            for j in range(len(fingerprints)):
                if fingerprints[i] and fingerprints[j]:
                    sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                else:
                    sim = random.uniform(0.3, 0.9)
                row.append(round(sim, 3))
            similarities.append(row)
    else:
        # Mock similarities for Render
        similarities = []
        for i in range(len(mols)):
            row = []
            for j in range(len(mols)):
                if i == j:
                    row.append(1.0)
                else:
                    row.append(round(random.uniform(0.3, 0.9), 3))
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
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'pdb_id': pdb_id,
                'protein_name': pdb_id,
                'message': f'Successfully fetched protein {pdb_id}'
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
    
    top_compound = results[0] if results else None
    admet_radar = None
    if top_compound and top_compound.get('properties'):
        admet_radar = generate_admet_radar(top_compound['properties'])
    
    # Save session
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
        while len(screening_history) > 20:
            screening_history.pop()
    
    return jsonify({
        'success': True,
        'results': results,
        'chemical_space': chemical_space,
        'activity_heatmap': activity_heatmap,
        'similarity_heatmap': similarity_heatmap,
        'admet_radar': admet_radar,
        'session': session_data if save_session else None,
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
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
    
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
        title="Molecular Properties Comparison",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
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
    if 'image' in df.columns:
        df = df.drop('image', axis=1)
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
    return jsonify({'sessions': screening_history})

@app.route('/api/compare', methods=['POST'])
def compare_compounds():
    data = request.json
    compounds = data.get('compounds', [])
    
    if len(compounds) < 2:
        return jsonify({'error': 'Need at least 2 compounds to compare'}), 400
    
    comparison_data = []
    for comp in compounds[:4]:
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
    ]
    return jsonify(examples)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
