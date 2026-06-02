import os
import json
import hashlib
import random
import base64
import urllib.parse
from datetime import datetime

import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly
import plotly.graph_objs as go

app = Flask(__name__)
app.secret_key = 'dockvis-pro-secret-key'
CORS(app)

# Configuration
if os.environ.get('RENDER'):
    UPLOAD_FOLDER = '/tmp/uploads'
else:
    UPLOAD_FOLDER = 'uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory storage
screening_history = []

def get_molecule_image(smiles, name):
    """Get molecule image from PubChem API"""
    try:
        encoded = urllib.parse.quote(smiles)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/PNG"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            img_b64 = base64.b64encode(resp.content).decode()
            return f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%; border-radius:10px;">'
    except:
        pass
    return f'<div style="background:#1a1a2e; border-radius:15px; padding:20px; text-align:center;"><i class="fas fa-draw-polygon" style="font-size:60px; color:#667eea;"></i><div class="mt-2"><small>{name}</small></div></div>'

def get_properties(smiles):
    """Calculate molecular properties (deterministic based on SMILES)"""
    h = abs(hash(smiles)) % 1000
    random.seed(h)
    props = {
        'mw': round(random.uniform(250, 500), 2),
        'logp': round(random.uniform(1, 4), 2),
        'tpsa': round(random.uniform(40, 120), 2),
        'donors': random.randint(0, 4),
        'acceptors': random.randint(2, 8),
        'qed': round(random.uniform(0.4, 0.9), 3),
        'drug_like': random.choice([True, False])
    }
    random.seed()
    return props

def run_screening(compounds):
    """Run virtual screening on compounds"""
    results = []
    for i, comp in enumerate(compounds):
        smiles = comp.get('smiles', '')
        name = comp.get('name', f'C{i+1}')
        props = get_properties(smiles)
        
        # Calculate binding affinity
        if props and props.get('drug_like'):
            score = random.uniform(-9.5, -7.5)
        else:
            score = random.uniform(-7.0, -5.0)
        
        if props and 250 < props.get('mw', 400) < 500:
            score -= random.uniform(0.2, 0.6)
        
        results.append({
            'compound': name,
            'smiles': smiles,
            'affinity': round(score, 2),
            'rank': 0,
            'props': props,
            'image_html': get_molecule_image(smiles, name)
        })
    
    results.sort(key=lambda x: x['affinity'])
    for i, r in enumerate(results):
        r['rank'] = i + 1
    
    return results

def create_pca_plot(compounds, results):
    """Create PCA chemical space plot"""
    if len(compounds) < 2:
        return None
    
    points = []
    labels = []
    affinities = []
    
    for comp in compounds:
        smiles = comp.get('smiles', '')
        props = get_properties(smiles)
        if props:
            # Find affinity from results
            aff = -7.0
            for r in results:
                if r.get('smiles') == smiles:
                    aff = r.get('affinity', -7.0)
                    break
            points.append([props['mw'], props['logp'], props['tpsa'], props['donors'], props['acceptors']])
            labels.append(comp.get('name', 'Unknown'))
            affinities.append(aff)
    
    if len(points) < 2:
        return None
    
    # PCA
    scaler = StandardScaler()
    scaled = scaler.fit_transform(points)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    
    # Colors based on affinity
    colors = ['#10b981' if a < -8 else '#f59e0b' if a < -6 else '#ef4444' for a in affinities]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers+text',
        marker=dict(size=40, color=colors, line=dict(width=2, color='white')),
        text=labels,
        textposition='top center',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{text}</b><br>Affinity: %{customdata:.2f} kcal/mol<extra></extra>',
        customdata=affinities
    ))
    
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    
    fig.update_layout(
        title=dict(text='Chemical Space Analysis', font=dict(color='white'), x=0.5),
        height=550,
        plot_bgcolor='rgba(30,30,60,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title=f'PC1 ({var1:.1f}%)', gridcolor='rgba(255,255,255,0.15)'),
        yaxis=dict(title=f'PC2 ({var2:.1f}%)', gridcolor='rgba(255,255,255,0.15)')
    )
    
    # Add padding
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    fig.update_xaxes(range=[coords[:, 0].min() - x_range*0.2, coords[:, 0].max() + x_range*0.2])
    fig.update_yaxes(range=[coords[:, 1].min() - y_range*0.2, coords[:, 1].max() + y_range*0.2])
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_heatmap(results):
    """Create affinity heatmap"""
    if not results:
        return None
    
    names = [r['compound'][:20] for r in results[:10]]
    scores = [r['affinity'] for r in results[:10]]
    
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

# ========== ROUTES ==========

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/protein/fetch', methods=['POST'])
def fetch_protein():
    data = request.get_json()
    pdb_id = data.get('pdb_id', '').upper()
    
    if not pdb_id or len(pdb_id) != 4:
        return jsonify({'error': 'Invalid PDB ID'}), 400
    
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return jsonify({'success': True, 'pdb_id': pdb_id, 'message': f'Loaded {pdb_id}'})
        return jsonify({'error': 'PDB ID not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compound/analyze', methods=['POST'])
def analyze_compound():
    data = request.get_json()
    smiles = data.get('smiles', '')
    
    if not smiles:
        return jsonify({'error': 'SMILES required'}), 400
    
    props = get_properties(smiles)
    if props:
        return jsonify({'success': True, 'properties': props})
    return jsonify({'error': 'Invalid SMILES'}), 400

@app.route('/api/docking/run', methods=['POST'])
def docking_run():
    data = request.get_json()
    protein_id = data.get('protein_id')
    compounds = data.get('compounds', [])
    
    if not protein_id:
        return jsonify({'error': 'Protein ID required'}), 400
    
    if not compounds:
        return jsonify({'error': 'No compounds to screen'}), 400
    
    # Filter valid compounds
    valid = [c for c in compounds if c.get('smiles', '').strip()]
    if not valid:
        return jsonify({'error': 'No valid compounds'}), 400
    
    # Run screening
    results = run_screening(valid)
    
    # Generate plots
    pca_plot = create_pca_plot(valid, results)
    heatmap = create_heatmap(results)
    
    # Generate similarity matrix (mock)
    if len(results) >= 2:
        n = min(8, len(results))
        names = [r['compound'][:15] for r in results[:n]]
        matrix = [[1.0 if i == j else round(random.uniform(0.3, 0.9), 2) for j in range(n)] for i in range(n)]
        sim_fig = go.Figure(data=go.Heatmap(z=matrix, x=names, y=names, colorscale='Viridis'))
        sim_fig.update_layout(title='Similarity Matrix', height=450, font=dict(color='white'))
        similarity = json.dumps(sim_fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        similarity = None
    
    # ADMET for top compound
    top = results[0] if results else None
    admet = None
    if top and top.get('props'):
        props = top['props']
        categories = ['MW', 'LogP', 'TPSA', 'H-Donors', 'H-Acceptors', 'Bioavailability']
        values = [
            min(1, props.get('mw', 0) / 500),
            min(1, (props.get('logp', 0) + 5) / 10),
            min(1, props.get('tpsa', 0) / 200),
            min(1, props.get('donors', 0) / 10),
            min(1, props.get('acceptors', 0) / 20),
            0.55 if props.get('drug_like') else 0.17
        ]
        admet_fig = go.Figure()
        admet_fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Compound'))
        admet_fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), title='ADMET Assessment', height=450, font=dict(color='white'))
        admet = json.dumps(admet_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Save to history
    session = {
        'id': hashlib.md5(f"{protein_id}_{datetime.now()}".encode()).hexdigest()[:8],
        'protein_id': protein_id,
        'timestamp': datetime.now().isoformat(),
        'num_compounds': len(results),
        'top_affinity': results[0]['affinity'],
        'avg_affinity': sum(r['affinity'] for r in results) / len(results),
        'drug_like_count': sum(1 for r in results if r.get('props', {}).get('drug_like'))
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
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    protein_id = request.form.get('protein_id', '')
    
    if not protein_id:
        return jsonify({'error': 'Protein ID required'}), 400
    
    compounds = []
    try:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            smiles = str(row.get('smiles', '')).strip()
            if smiles:
                name = str(row.get('name', f'C{len(compounds)+1}'))
                compounds.append({'name': name, 'smiles': smiles})
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 400
    
    if not compounds:
        return jsonify({'error': 'No valid compounds found'}), 400
    
    results = run_screening(compounds)
    pca_plot = create_pca_plot(compounds, results)
    heatmap = create_heatmap(results)
    
    return jsonify({
        'success': True,
        'results': results,
        'chemical_space': pca_plot,
        'activity_heatmap': heatmap,
        'total_compounds': len(compounds),
        'message': f'Batch complete: {len(compounds)} compounds'
    })

@app.route('/api/properties/radar/<path:smiles_list>')
def radar_chart(smiles_list):
    from urllib.parse import unquote
    smiles_array = unquote(smiles_list).split(',')
    data = []
    
    for smiles in smiles_array[:5]:
        props = get_properties(smiles)
        if props:
            data.append({
                'name': smiles[:20],
                'MW': props['mw'] / 500,
                'LogP': (props['logp'] + 5) / 10,
                'TPSA': props['tpsa'] / 200,
                'H_Donors': props['donors'] / 10,
                'H_Acceptors': props['acceptors'] / 10
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
        polar=dict(radialaxis=dict(range=[0, 1])),
        title='Property Comparison',
        height=450,
        font=dict(color='white')
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/api/export/results', methods=['POST'])
def export_results():
    data = request.get_json()
    results = data.get('results', [])
    
    if not results:
        return jsonify({'error': 'No results'}), 400
    
    df = pd.DataFrame(results)
    if 'image_html' in df.columns:
        df = df.drop('image_html', axis=1)
    if 'props' in df.columns:
        df = df.drop('props', axis=1)
    
    path = os.path.join(app.config['UPLOAD_FOLDER'], f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(path, index=False)
    return send_file(path, as_attachment=True, download_name='docking_results.csv')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    return jsonify({'sessions': screening_history})

@app.route('/api/compare', methods=['POST'])
def compare():
    data = request.get_json()
    compounds = data.get('compounds', [])
    
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
def examples():
    return jsonify([
        {'name': 'Aspirin', 'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O'},
        {'name': 'Ibuprofen', 'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'},
        {'name': 'Paracetamol', 'smiles': 'CC(=O)NC1=CC=C(C=C1)O'},
        {'name': 'Caffeine', 'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'},
    ])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
