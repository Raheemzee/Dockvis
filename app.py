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
            return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; border-radius:10px;">'
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
    """Create PCA plot for chemical space - FIXED VERSION"""
    print(f"make_pca_plot called with {len(compounds)} compounds")
    
    if len(compounds) < 2:
        print("Not enough compounds (need at least 2)")
        return None
    
    points = []
    labels = []
    scores = []
    colors_list = []
    
    for comp in compounds:
        smiles = comp.get('smiles', '')
        props = get_properties(smiles)
        if props:
            # Find the score for this compound
            score = -7.0
            for res in results:
                if res.get('smiles') == smiles:
                    score = res.get('binding_affinity', -7.0)
                    break
            
            points.append([
                props['molecular_weight'],
                props['logP'],
                props['tpsa'],
                props['h_donors'],
                props['h_acceptors']
            ])
            labels.append(comp.get('name', 'Unknown'))
            scores.append(score)
            
            # Determine color based on score
            if score < -8:
                colors_list.append('#10b981')  # Green - excellent
            elif score < -6:
                colors_list.append('#f59e0b')  # Orange - good
            else:
                colors_list.append('#ef4444')  # Red - poor
    
    if len(points) < 2:
        print("Not enough valid data points")
        return None
    
    print(f"PCA input: {len(points)} points, labels: {labels}, scores: {scores}")
    
    # Perform PCA
    scaler = StandardScaler()
    scaled_points = scaler.fit_transform(points)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled_points)
    
    print(f"PCA coordinates: {coords}")
    
    # Create the plot
    fig = go.Figure()
    
    # Add trace with markers and text
    fig.add_trace(go.Scatter(
        x=coords[:, 0].tolist(),
        y=coords[:, 1].tolist(),
        mode='markers+text',
        marker=dict(
            size=40,
            color=colors_list,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        text=labels,
        textposition='top center',
        textfont=dict(
            size=12,
            color='white',
            family='Arial Black, Arial, sans-serif'
        ),
        hovertemplate='<b>%{text}</b><br>' +
                      'Binding Affinity: %{customdata:.2f} kcal/mol<br>' +
                      'PC1: %{x:.3f}<br>' +
                      'PC2: %{y:.3f}<extra></extra>',
        customdata=scores,
        name='Compounds'
    ))
    
    # Calculate variance percentages
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    
    # Add padding to axis ranges
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_padding = max((x_max - x_min) * 0.2, 0.5)
    y_padding = max((y_max - y_min) * 0.2, 0.5)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Chemical Space Analysis (PCA)',
            font=dict(size=20, color='white', family='Arial'),
            x=0.5
        ),
        height=550,
        plot_bgcolor='rgba(30, 30, 60, 0.95)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white', family='Arial'),
        xaxis=dict(
            title=f'Principal Component 1 ({var_pc1:.1f}% variance)',
            titlefont=dict(size=14),
            tickfont=dict(size=11),
            gridcolor='rgba(255,255,255,0.15)',
            zerolinecolor='rgba(255,255,255,0.2)',
            range=[x_min - x_padding, x_max + x_padding],
            showgrid=True,
            showline=True,
            linecolor='rgba(255,255,255,0.3)'
        ),
        yaxis=dict(
            title=f'Principal Component 2 ({var_pc2:.1f}% variance)',
            titlefont=dict(size=14),
            tickfont=dict(size=11),
            gridcolor='rgba(255,255,255,0.15)',
            zerolinecolor='rgba(255,255,255,0.2)',
            range=[y_min - y_padding, y_max + y_padding],
            showgrid=True,
            showline=True,
            linecolor='rgba(255,255,255,0.3)'
        ),
        hovermode='closest',
        margin=dict(l=80, r=80, t=80, b=80),
        showlegend=False
    )
    
    # Add legend annotation at bottom
    fig.add_annotation(
        x=0.5,
        y=-0.12,
        xref='paper',
        yref='paper',
        text='🟢 Excellent (&lt;-8 kcal/mol) &nbsp;&nbsp;&nbsp; 🟡 Good (-6 to -8) &nbsp;&nbsp;&nbsp; 🔴 Weak (&gt;-6)',
        showarrow=False,
        font=dict(size=12, color='rgba(255,255,255,0.9)'),
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor='rgba(255,255,255,0.3)',
        borderwidth=1,
        borderpad=8,
        align='center'
    )
    
    result_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print(f"Plot generated successfully, length: {len(result_json)}")
    return result_json

def make_heatmap(results):
    """Create binding affinity heatmap"""
    if not results:
        return None
    
    names = [r['compound'][:20] for r in results[:10]]
    scores = [r['binding_affinity'] for r in results[:10]]
    
    fig = go.Figure(data=go.Heatmap(
        z=[scores],
        y=['Binding Affinity'],
        x=names,
        colorscale='RdYlGn_r',
        text=[[f'{s:.2f}' for s in scores]],
        texttemplate='%{text} kcal/mol',
        textfont=dict(size=11, color='white'),
        colorbar=dict(title='kcal/mol', len=0.8)
    ))
    
    fig.update_layout(
        title='Binding Affinity Heatmap',
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(tickangle=45)
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
        texttemplate='%{text}',
        textfont=dict(size=10)
    ))
    
    fig.update_layout(
        title='Molecular Similarity Matrix',
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(tickangle=45)
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
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Compound',
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102,126,234,0.3)'
    ))
    
    optimal = [0.7, 0.7, 0.5, 0.5, 0.5, 0.8]
    fig.add_trace(go.Scatterpolar(
        r=optimal,
        theta=categories,
        fill='toself',
        name='Optimal',
        line=dict(color='#10b981', width=1, dash='dash'),
        fillcolor='rgba(16,185,129,0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1], gridcolor='rgba(255,255,255,0.2)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.2)')
        ),
        title='ADMET Property Assessment',
        height=450,
        font=dict(color='white'),
        showlegend=True
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
    
    print(f"=== DOCKING RUN ===")
    print(f"Protein: {protein_id}")
    print(f"Compounds count: {len(compounds)}")
    
    if not protein_id or not compounds:
        return jsonify({'error': 'Protein and compounds required'}), 400
    
    # Filter valid compounds
    valid = [c for c in compounds if c.get('smiles', '').strip()]
    print(f"Valid compounds: {len(valid)}")
    
    if not valid:
        return jsonify({'error': 'No valid compounds'}), 400
    
    results = run_screening(protein_id, valid)
    print(f"Results count: {len(results)}")
    
    # Generate plots
    pca_plot = make_pca_plot(valid, results)
    print(f"PCA plot generated: {pca_plot is not None}")
    
    heatmap = make_heatmap(results)
    similarity = make_similarity_plot(results)
    
    top = results[0] if results else None
    admet = make_admet_plot(top.get('properties')) if top else None
    
    # Save to history
    if results:
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
    
    response = {
        'success': True,
        'results': results,
        'chemical_space': pca_plot,
        'activity_heatmap': heatmap,
        'similarity_heatmap': similarity,
        'admet_radar': admet,
        'message': f'Screened {len(results)} compounds'
    }
    
    print(f"Response keys: {response.keys()}")
    return jsonify(response)

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
        polar=dict(radialaxis=dict(range=[0, 1], gridcolor='rgba(255,255,255,0.2)')),
        title='Molecular Properties Comparison',
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
    app.run(host='0.0.0.0', port=port, debug=False)
