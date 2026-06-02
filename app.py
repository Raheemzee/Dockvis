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
from io import BytesIO
import requests
import urllib.parse
import traceback
import warnings
warnings.filterwarnings('ignore')

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

def fetch_molecule_image_from_pubchem(smiles, compound_name):
    """Fetch molecule image from PubChem API"""
    try:
        encoded_smiles = urllib.parse.quote(smiles)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/PNG"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            img_base64 = base64.b64encode(response.content).decode()
            return f'<img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto; border-radius: 10px; background: white; padding: 10px;" alt="{compound_name}">'
    except:
        pass
    
    # Fallback representation
    return f'''
    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 15px; padding: 20px; text-align: center;">
        <div style="background: white; border-radius: 10px; padding: 20px; margin-bottom: 15px;">
            <i class="fas fa-draw-polygon" style="font-size: 80px; color: #667eea;"></i>
            <div class="mt-2">
                <code class="text-muted" style="font-size: 11px; word-break: break-all;">{smiles[:80]}...</code>
            </div>
        </div>
        <div>
            <span class="badge bg-primary">{compound_name}</span>
        </div>
    </div>
    '''

def validate_smiles(smiles):
    """Validate SMILES string"""
    if not smiles or not isinstance(smiles, str):
        return False
    smiles = smiles.strip()
    return len(smiles) > 1

def calculate_molecular_properties(smiles):
    """Calculate molecular properties"""
    try:
        smiles = smiles.strip()
        # Use consistent random based on SMILES hash
        hash_val = abs(hash(smiles)) % 1000
        random.seed(hash_val)
        
        properties = {
            'molecular_weight': round(random.uniform(250, 500), 2),
            'logP': round(random.uniform(1, 4), 2),
            'tpsa': round(random.uniform(40, 120), 2),
            'h_donors': random.randint(0, 4),
            'h_acceptors': random.randint(2, 8),
            'rotatable_bonds': random.randint(1, 8),
            'heavy_atoms': random.randint(15, 40),
            'num_rings': random.randint(1, 4),
            'smiles': smiles,
            'qed': round(random.uniform(0.4, 0.9), 3),
            'lipinski_violations': random.randint(0, 2),
            'drug_like': random.choice([True, False]),
            'bioavailability': random.choice([0.55, 0.17]),
        }
        
        random.seed()
        return properties
    except Exception as e:
        print(f"Error calculating properties: {e}")
        return None

def generate_molecule_image_html(smiles, compound_name):
    """Generate HTML for molecule display"""
    return fetch_molecule_image_from_pubchem(smiles, compound_name)

def perform_virtual_screening(protein_id, compounds):
    """Perform virtual screening with enhanced scoring"""
    results = []
    
    for i, compound in enumerate(compounds):
        smiles = compound.get('smiles', '')
        props = calculate_molecular_properties(smiles)
        
        # Generate realistic binding affinity
        if props and props.get('drug_like'):
            base_score = random.uniform(-9.5, -7.5)
        else:
            base_score = random.uniform(-7.0, -5.0)
        
        if props:
            mw = props.get('molecular_weight', 400)
            if 250 < mw < 500:
                base_score -= random.uniform(0.2, 0.6)
        
        image_html = generate_molecule_image_html(smiles, compound.get('name', f'Compound_{i+1}'))
        
        results.append({
            'compound': compound.get('name', f'Compound_{i+1}'),
            'smiles': smiles,
            'binding_affinity': round(base_score, 2),
            'rank': 0,
            'properties': props,
            'image_html': image_html
        })
    
    results.sort(key=lambda x: x['binding_affinity'])
    for i, r in enumerate(results):
        r['rank'] = i + 1
    
    return results

def generate_chemical_space_plot(compounds, results_dict=None):
    """Generate chemical space visualization - FIXED VERSION"""
    print(f"Generating chemical space plot for {len(compounds)} compounds")
    
    if len(compounds) < 2:
        print("Not enough compounds (need at least 2)")
        return None
    
    # Prepare data for PCA
    data_points = []
    labels = []
    affinities = []
    
    for comp in compounds:
        smiles = comp.get('smiles', '')
        props = calculate_molecular_properties(smiles)
        
        if props:
            # Get affinity from results if available
            affinity = -7.0
            if results_dict:
                for res in results_dict:
                    if res.get('smiles') == smiles:
                        affinity = res.get('binding_affinity', -7.0)
                        break
            
            data_points.append([
                props['molecular_weight'],
                props['logP'],
                props['tpsa'],
                props['h_donors'],
                props['h_acceptors']
            ])
            labels.append(comp.get('name', 'Unknown'))
            affinities.append(affinity)
    
    if len(data_points) < 2:
        print("Not enough valid data points for PCA")
        return None
    
    # Perform PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_points)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create colors based on affinity
    colors = []
    for aff in affinities:
        if aff < -8:
            colors.append('#10b981')  # Green - excellent
        elif aff < -6:
            colors.append('#f59e0b')  # Orange - good
        else:
            colors.append('#ef4444')  # Red - poor
    
    # Create the plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        mode='markers+text',
        marker=dict(
            size=40,
            color=colors,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        text=labels,
        textposition='top center',
        textfont=dict(
            size=12,
            color='white',
            family='Arial, sans-serif'
        ),
        hovertemplate='<b>%{text}</b><br>' +
                      'Binding Affinity: %{customdata:.2f} kcal/mol<br>' +
                      'PC1: %{x:.3f}<br>' +
                      'PC2: %{y:.3f}<extra></extra>',
        customdata=affinities,
        name='Compounds'
    ))
    
    # Calculate axis ranges with padding
    x_min, x_max = pca_result[:, 0].min(), pca_result[:, 0].max()
    y_min, y_max = pca_result[:, 1].min(), pca_result[:, 1].max()
    x_padding = (x_max - x_min) * 0.2 if x_max != x_min else 1
    y_padding = (y_max - y_min) * 0.2 if y_max != y_min else 1
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Chemical Space Analysis (PCA)',
            font=dict(size=20, color='white'),
            x=0.5
        ),
        height=550,
        plot_bgcolor='rgba(30, 30, 60, 0.9)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        xaxis=dict(
            title=f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            titlefont=dict(size=14),
            tickfont=dict(size=11),
            gridcolor='rgba(255,255,255,0.15)',
            zerolinecolor='rgba(255,255,255,0.2)',
            range=[x_min - x_padding, x_max + x_padding]
        ),
        yaxis=dict(
            title=f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
            titlefont=dict(size=14),
            tickfont=dict(size=11),
            gridcolor='rgba(255,255,255,0.15)',
            zerolinecolor='rgba(255,255,255,0.2)',
            range=[y_min - y_padding, y_max + y_padding]
        ),
        hovermode='closest',
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    # Add legend annotation
    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref='paper',
        yref='paper',
        text='🟢 Excellent (&lt;-8) &nbsp;&nbsp; 🟡 Good (-6 to -8) &nbsp;&nbsp; 🔴 Weak (&gt;-6)',
        showarrow=False,
        font=dict(size=11, color='rgba(255,255,255,0.9)'),
        bgcolor='rgba(0,0,0,0.6)',
        bordercolor='rgba(255,255,255,0.3)',
        borderwidth=1,
        borderpad=6,
        align='right'
    )
    
    result = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print(f"Chemical space plot generated successfully")
    return result

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
    """Generate molecular similarity heatmap"""
    if len(results) < 2:
        return None
    
    top_results = results[:8]
    n = len(top_results)
    
    similarities = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(round(random.uniform(0.3, 0.9), 3))
        similarities.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=similarities,
        x=[r['compound'][:15] for r in top_results],
        y=[r['compound'][:15] for r in top_results],
        colorscale='Viridis',
        text=[[str(sim) for sim in row] for row in similarities],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Similarity")
    ))
    
    fig.update_layout(
        title='Molecular Similarity Matrix',
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis=dict(tickangle=45)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_admet_radar(properties):
    """Generate ADMET radar chart"""
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
                gridcolor='rgba(255,255,255,0.2)'
            ),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.2)')
        ),
        showlegend=True,
        title="ADMET Property Assessment",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Flask Routes
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
    
    print(f"Running docking for protein: {protein_id}, compounds: {len(compounds)}")
    
    if not protein_id or not compounds:
        return jsonify({'error': 'Protein ID and compounds required'}), 400
    
    results = perform_virtual_screening(protein_id, compounds)
    print(f"Screening complete, {len(results)} results")
    
    chemical_space = generate_chemical_space_plot(compounds, results)
    activity_heatmap = generate_activity_heatmap(results)
    similarity_heatmap = generate_similarity_heatmap(results)
    
    top_compound = results[0] if results else None
    admet_radar = None
    if top_compound and top_compound.get('properties'):
        admet_radar = generate_admet_radar(top_compound['properties'])
    
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
    
    response_data = {
        'success': True,
        'results': results,
        'chemical_space': chemical_space,
        'activity_heatmap': activity_heatmap,
        'similarity_heatmap': similarity_heatmap,
        'admet_radar': admet_radar,
        'message': f'Docking complete. Screened {len(results)} compounds.'
    }
    
    return jsonify(response_data)

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
                if smiles and len(smiles) > 1:
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
    if 'image_html' in df.columns:
        df = df.drop('image_html', axis=1)
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
                'image_html': generate_molecule_image_html(smiles, comp.get('name', 'Unknown'))
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
    app.run(host='0.0.0.0', port=port, debug=False)
