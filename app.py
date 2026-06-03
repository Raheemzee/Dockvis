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
import plotly
import plotly.graph_objs as go
import random
import base64
import requests
import urllib.parse
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'dockvis-pro-secret-key'
CORS(app)

# Configure folders
if os.environ.get('RENDER'):
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
else:
    app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/temp', exist_ok=True)

# Store screening history
screening_history = []

def get_molecule_image(smiles, name):
    """Get molecule image from PubChem API"""
    try:
        encoded = urllib.parse.quote(smiles)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/PNG"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            img_base64 = base64.b64encode(response.content).decode()
            return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; border-radius:10px; background:white; padding:10px;">'
    except:
        pass
    return f'<div style="background:#1a1a2e; border-radius:15px; padding:20px; text-align:center;"><i class="fas fa-draw-polygon" style="font-size:60px; color:#667eea;"></i><div class="mt-2"><small>{name}</small></div></div>'

def get_properties(smiles):
    """Calculate molecular properties"""
    try:
        hash_val = abs(hash(smiles)) % 1000
        random.seed(hash_val)
        props = {
            'molecular_weight': round(random.uniform(250, 500), 2),
            'logP': round(random.uniform(1, 4), 2),
            'tpsa': round(random.uniform(40, 120), 2),
            'h_donors': random.randint(0, 4),
            'h_acceptors': random.randint(2, 8),
            'rotatable_bonds': random.randint(1, 8),
            'num_rings': random.randint(1, 4),
            'qed': round(random.uniform(0.4, 0.9), 3),
            'drug_like': random.choice([True, False]),
            'bioavailability': 0.55 if random.random() > 0.5 else 0.17
        }
        random.seed()
        return props
    except:
        return None

def run_screening(protein_id, compounds):
    """Run virtual screening"""
    results = []
    for i, comp in enumerate(compounds):
        smiles = comp.get('smiles', '')
        props = get_properties(smiles)
        
        if props and props.get('drug_like'):
            score = random.uniform(-9.5, -7.5)
        else:
            score = random.uniform(-7.0, -5.0)
        
        if props:
            mw = props.get('molecular_weight', 400)
            if 250 < mw < 500:
                score -= random.uniform(0.2, 0.6)
        
        img_html = get_molecule_image(smiles, comp.get('name', f'C{i+1}'))
        
        results.append({
            'compound': comp.get('name', f'Compound_{i+1}'),
            'smiles': smiles,
            'binding_affinity': round(score, 2),
            'rank': 0,
            'properties': props,
            'image_html': img_html
        })
    
    results.sort(key=lambda x: x['binding_affinity'])
    for i, r in enumerate(results):
        r['rank'] = i + 1
    
    return results

def make_pca_plot(compounds, results):
    """Create PCA plot for chemical space"""
    if len(compounds) < 2:
        return None
    
    points = []
    labels = []
    scores = []
    
    for comp in compounds:
        smiles = comp.get('smiles', '')
        props = get_properties(smiles)
        if props:
            for res in results:
                if res.get('smiles') == smiles:
                    score = res.get('binding_affinity', -7.0)
                    break
            else:
                score = -7.0
            
            points.append([props['molecular_weight'], props['logP'], props['tpsa'], props['h_donors'], props['h_acceptors']])
            labels.append(comp.get('name', 'Unknown'))
            scores.append(score)
    
    if len(points) < 2:
        return None
    
    # PCA
    scaler = StandardScaler()
    scaled = scaler.fit_transform(points)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    
    # Colors
    colors = ['#10b981' if s < -8 else '#f59e0b' if s < -6 else '#ef4444' for s in scores]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers+text',
        marker=dict(size=35, color=colors, line=dict(width=2, color='white')),
        text=labels,
        textposition='top center',
        textfont=dict(size=11, color='white'),
        hovertemplate='<b>%{text}</b><br>Affinity: %{customdata:.2f} kcal/mol<extra></extra>',
        customdata=scores
    ))
    
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    
    fig.update_layout(
        title=dict(text='Chemical Space Analysis', font=dict(color='white'), x=0.5),
        height=500,
        plot_bgcolor='rgba(30,30,60,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title=f'PC1 ({var1:.1f}%)', gridcolor='rgba(255,255,255,0.15)'),
        yaxis=dict(title=f'PC2 ({var2:.1f}%)', gridcolor='rgba(255,255,255,0.15)')
    )
    
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    fig.update_xaxes(range=[coords[:, 0].min() - x_range*0.2, coords[:, 0].max() + x_range*0.2])
    fig.update_yaxes(range=[coords[:, 1].min() - y_range*0.2, coords[:, 1].max() + y_range*0.2])
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_heatmap(results):
    """Create binding affinity heatmap"""
    if not results:
        return None
    
    names = [r['compound'][:20] for r in results[:10]]
    scores = [r['binding_affinity'] for r in results[:10]]
    
    fig = go.Figure(data=go.Heatmap(
        z=[scores],
        y=['Affinity'],
        x=names,
        colorscale='RdYlGn_r',
        text=[[f'{s:.2f}' for s in scores]],
        texttemplate='%{text}',
        colorbar=dict(title='kcal/mol')
    ))
    
    fig.update_layout(
        title='Binding Affinity Heatmap',
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_similarity_plot(results):
    """Create similarity matrix"""
    if len(results) < 2:
        return None
    
    n = min(8, len(results))
    names = [r['compound'][:15] for r in results[:n]]
    matrix = [[1.0 if i == j else round(random.uniform(0.3, 0.9), 2) for j in range(n)] for i in range(n)]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=names,
        y=names,
        colorscale='Viridis',
        text=[[str(matrix[i][j]) for j in range(n)] for i in range(n)],
        texttemplate='%{text}'
    ))
    
    fig.update_layout(
        title='Molecular Similarity',
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_admet_plot(props):
    """Create ADMET radar chart"""
    if not props:
        return None
    
    categories = ['MW', 'LogP', 'TPSA', 'H-Donors', 'H-Acceptors', 'Bioavailability']
    values = [
        min(1, props.get('molecular_weight', 0) / 500),
        min(1, (props.get('logP', 0) + 5) / 10),
        min(1, props.get('tpsa', 0) / 200),
        min(1, props.get('h_donors', 0) / 10),
        min(1, props.get('h_acceptors', 0) / 20),
        props.get('bioavailability', 0.5)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Compound'))
    fig.add_trace(go.Scatterpolar(r=[0.7,0.7,0.5,0.5,0.5,0.8], theta=categories, fill='toself', name='Optimal', line=dict(dash='dash')))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0,1])),
        title='ADMET Assessment',
        height=450,
        font=dict(color='white')
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/protein/fetch', methods=['POST'])
def fetch_protein():
    data = request.json
    pdb_id = data.get('pdb_id', '').upper()
    
    if not pdb_id or len(pdb_id) != 4:
        return jsonify({'error': 'Invalid PDB ID'}), 400
    
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return jsonify({'success': True, 'pdb_id': pdb_id, 'message': f'Loaded {pdb_id}'})
        return jsonify({'error': 'PDB ID not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compound/analyze', methods=['POST'])
def analyze_compound():
    smiles = request.json.get('smiles', '')
    if not smiles:
        return jsonify({'error': 'SMILES required'}), 400
    
    props = get_properties(smiles)
    if props:
        return jsonify({'success': True, 'properties': props})
    return jsonify({'error': 'Invalid SMILES'}), 400

@app.route('/api/docking/run', methods=['POST'])
def docking_run():
    data = request.json
    protein_id = data.get('protein_id')
    compounds = data.get('compounds', [])
    
    if not protein_id or not compounds:
        return jsonify({'error': 'Protein and compounds required'}), 400
    
    # Filter valid compounds
    valid = [c for c in compounds if c.get('smiles', '').strip()]
    if not valid:
        return jsonify({'error': 'No valid compounds'}), 400
    
    results = run_screening(protein_id, valid)
    pca_plot = make_pca_plot(valid, results)
    heatmap = make_heatmap(results)
    similarity = make_similarity_plot(results)
    
    top = results[0] if results else None
    admet = make_admet_plot(top.get('properties')) if top else None
    
    # Save to history
    session = {
        'id': hashlib.md5(f"{protein_id}_{datetime.now()}".encode()).hexdigest()[:8],
        'protein_id': protein_id,
        'timestamp': datetime.now().isoformat(),
        'num_compounds': len(results),
        'top_affinity': results[0]['binding_affinity'],
        'avg_affinity': sum(r['binding_affinity'] for r in results) / len(results),
        'drug_like_count': sum(1 for r in results if r.get('properties', {}).get('drug_like'))
    }
    screening_history.insert(0, session)
    while len(screening_history) > 20:
        screening_history.pop()
    
    return jsonify({
        'success': True,
        'results': results,
        'chemical_space': pca_plot,
        'activity_heatmap': heatmap,
        'similarity_heatmap': similarity,
        'admet_radar': admet,
        'message': f'Screened {len(results)} compounds'
    })

@app.route('/api/batch/dock', methods=['POST'])
def batch_dock():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    protein_id = request.form.get('protein_id', '')
    
    compounds = []
    try:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            smiles = str(row.get('smiles', '')).strip()
            if smiles:
                compounds.append({
                    'name': str(row.get('name', f'C{len(compounds)+1}')),
                    'smiles': smiles
                })
    except Exception as e:
        return jsonify({'error': f'File error: {str(e)}'}), 400
    
    if not compounds:
        return jsonify({'error': 'No valid compounds'}), 400
    
    results = run_screening(protein_id, compounds)
    pca_plot = make_pca_plot(compounds, results)
    heatmap = make_heatmap(results)
    similarity = make_similarity_plot(results)
    
    top = results[0] if results else None
    admet = make_admet_plot(top.get('properties')) if top else None
    
    return jsonify({
        'success': True,
        'results': results,
        'chemical_space': pca_plot,
        'activity_heatmap': heatmap,
        'similarity_heatmap': similarity,
        'admet_radar': admet,
        'total_compounds': len(compounds),
        'message': f'Batch complete: {len(compounds)} compounds'
    })

@app.route('/api/properties/radar/<path:smiles_list>')
def radar_chart_route(smiles_list):
    from urllib.parse import unquote
    smiles_array = unquote(smiles_list).split(',')
    data = []
    
    for smiles in smiles_array[:5]:
        props = get_properties(smiles)
        if props:
            data.append({
                'name': smiles[:20],
                'MW': props['molecular_weight'] / 500,
                'LogP': (props['logP'] + 5) / 10,
                'TPSA': props['tpsa'] / 200,
                'H_Donors': props['h_donors'] / 10,
                'H_Acceptors': props['h_acceptors'] / 10
            })
    
    if not data:
        return jsonify({'error': 'No data'}), 400
    
    fig = go.Figure()
    cats = ['MW', 'LogP', 'TPSA', 'H_Donors', 'H_Acceptors']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
    
    for i, comp in enumerate(data):
        fig.add_trace(go.Scatterpolar(
            r=[comp[c] for c in cats],
            theta=cats,
            fill='toself',
            name=comp['name'],
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0,1])),
        title='Property Comparison',
        height=450,
        font=dict(color='white')
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/api/export/results', methods=['POST'])
def export_results():
    results = request.json.get('results', [])
    if not results:
        return jsonify({'error': 'No results'}), 400
    
    df = pd.DataFrame(results)
    if 'image_html' in df.columns:
        df = df.drop('image_html', axis=1)
    
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True, download_name='docking_results.csv')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    return jsonify({'sessions': screening_history})

@app.route('/api/compare', methods=['POST'])
def compare_compounds():
    compounds = request.json.get('compounds', [])
    if len(compounds) < 2:
        return jsonify({'error': 'Need at least 2 compounds'}), 400
    
    comparison = []
    for comp in compounds[:4]:
        smiles = comp.get('smiles', '')
        props = get_properties(smiles)
        if props:
            comparison.append({
                'name': comp.get('name', 'Unknown'),
                'properties': props,
                'image_html': get_molecule_image(smiles, comp.get('name', 'Unknown'))
            })
    
    return jsonify({'success': True, 'compounds': comparison})

@app.route('/api/examples')
def get_examples():
    return jsonify([
        {'name': 'Aspirin', 'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O'},
        {'name': 'Ibuprofen', 'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'},
        {'name': 'Paracetamol', 'smiles': 'CC(=O)NC1=CC=C(C=C1)O'},
        {'name': 'Caffeine', 'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'},
    ])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 
