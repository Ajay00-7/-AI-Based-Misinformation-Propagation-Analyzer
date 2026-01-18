"""
Visualization Module for Misinformation Detection System
Generates all required charts and graphs for the web interface
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import base64
from io import BytesIO
import networkx as nx

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')

class ChartGenerator:
    def __init__(self):
        """Initialize chart generator"""
        self.colors = {
            'fake': '#FF6B9D',  # Pink
            'real': '#4ECDC4',  # Teal
            'susceptible': '#95E1D3',  # Light green
            'infected': '#F38181',  # Coral
            'recovered': '#AA96DA',  # Purple
            'bert': '#6C5CE7',  # Purple
            'svm': '#00D2D3'  # Cyan
        }
    
    def pie_chart_fake_vs_real(self, fake_count, real_count):
        """Generate interactive pie chart for Fake vs Real"""
        labels = ['Fake News', 'Real News']
        values = [fake_count, real_count]
        colors = [self.colors['fake'], self.colors['real']]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=.3,
            marker=dict(colors=colors, line=dict(color='#fff', width=2)),
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )])
        
        fig.update_layout(
            title='Fake vs Real News Distribution',
            title_x=0.5,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fff') # Assuming dark theme
        )
        return fig.to_json()
    
    def bar_chart_model_comparison(self, bert_accuracy, svm_accuracy):
        """Generate interactive bar chart"""
        models = ['BERT', 'SVM']
        accuracies = [bert_accuracy * 100, svm_accuracy * 100]
        colors = [self.colors['bert'], self.colors['svm']]
        
        fig = go.Figure(data=[go.Bar(
            x=models, y=accuracies,
            marker_color=colors,
            text=[f'{acc:.1f}%' for acc in accuracies],
            textposition='auto',
        )])
        
        fig.update_layout(
            title='Model Performance Comparison',
            title_x=0.5,
            yaxis=dict(title='Accuracy (%)', range=[0, 100]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fff')
        )
        return fig.to_json()
    
    def line_chart_processing_pipeline(self):
        """Generate interactive pipeline chart"""
        stages = ['Input', 'Preprocessing', 'Tokenization', 'BERT Embedding', 'Classification', 'Output']
        complexity = [10, 30, 50, 85, 95, 100]
        
        fig = go.Figure(data=go.Scatter(
            x=stages, y=complexity,
            mode='lines+markers',
            line=dict(color=self.colors['bert'], width=4),
            marker=dict(size=12, color='white', line=dict(width=2, color=self.colors['bert']))
        ))
        
        fig.update_layout(
            title='Information Processing Pipeline',
            title_x=0.5,
            yaxis=dict(title='Completion (%)', range=[0, 110]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fff')
        )
        return fig.to_json()
    
    def line_graph_sir_model(self, sir_data):
        """Generate interactive SIR model chart"""
        time = sir_data['time']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=sir_data['susceptible'], name='Susceptible', 
                               line=dict(color=self.colors['susceptible'], width=3)))
        fig.add_trace(go.Scatter(x=time, y=sir_data['infected'], name='Infected',
                               line=dict(color=self.colors['infected'], width=3)))
        fig.add_trace(go.Scatter(x=time, y=sir_data['recovered'], name='Recovered',
                               line=dict(color=self.colors['recovered'], width=3)))
        
        # Highlight peak
        I = sir_data['infected']
        peak_idx = np.argmax(I)
        fig.add_trace(go.Scatter(
            x=[time[peak_idx]], y=[I[peak_idx]],
            mode='markers', name='Peak Infection',
            marker=dict(color='red', size=15, symbol='star')
        ))
        
        fig.update_layout(
            title='SIR Model: Propagation Over Time',
            title_x=0.5,
            xaxis_title='Time (days)',
            yaxis_title='Population',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fff'),
            hovermode='x unified'
        )
        return fig.to_json()
    
    def network_graph_visualization(self, graph_data):
        """Generate 3D network graph visualization"""
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        edge_x = []
        edge_y = []
        edge_z = []
        
        # Nodes need 3D coordinates. If 2D provided, add random Z
        for node in nodes:
            if 'z' not in node:
                node['z'] = np.random.uniform(0, 10)
        
        for edge in edges:
            source_node = next(n for n in nodes if n['id'] == edge['source'])
            target_node = next(n for n in nodes if n['id'] == edge['target'])
            edge_x.extend([source_node['x'], target_node['x'], None])
            edge_y.extend([source_node['y'], target_node['y'], None])
            edge_z.extend([source_node['z'], target_node['z'], None])
            
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=1, color='#888'),
            opacity=0.5,
            mode='lines',
            hoverinfo='none'
        )
        
        node_x = [n['x'] for n in nodes]
        node_y = [n['y'] for n in nodes]
        node_z = [n['z'] for n in nodes]
        node_size = [n['size'] for n in nodes]
        node_color = [n['degree'] for n in nodes]
        node_text = [f'User {n["id"]}<br>Connections: {n["degree"]}' for n in nodes]
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                colorbar=dict(title='Connections'),
                line=dict(width=0)
            ),
            text=node_text,
            hoverinfo='text'
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title='Information Spread Network (3D)',
            title_x=0.5,
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fff'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig.to_json()
    
    def pie_chart_influencer_contribution(self, influencers):
        """Generate interactive influencer chart"""
        top_5 = influencers[:5]
        other_score = sum(inf['influence_score'] for inf in influencers[5:])
        
        labels = [f"User {inf['id']}" for inf in top_5] + ['Others']
        values = [inf['influence_score'] for inf in top_5] + [other_score]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            title='Top Influencers Contribution',
            title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fff')
        )
        return fig.to_json()
    
    def create_static_confidence_chart(self, bert_conf, svm_conf):
        """Create static matplotlib chart for PDF report"""
        fig, ax = plt.subplots(figsize=(6, 4))
        models = ['BERT', 'SVM']
        scores = [bert_conf * 100, svm_conf * 100]
        colors = [self.colors['bert'], self.colors['svm']]
        
        bars = ax.bar(models, scores, color=colors,width=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Confidence (%)')
        ax.set_title('Model Confidence Comparison')
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        # Save to temporary file
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        plt.savefig(path, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        return path
    
    def _fig_to_base64(self, fig):
        # Deprecated
        pass

if __name__ == "__main__":
    # Quick test to ensure syntax is correct
    cg = ChartGenerator()
    print("Testing 3D Chart Generation...")
    try:
        # Mock data for graph
        nodes = [{'id': i, 'x': np.random.rand(), 'y': np.random.rand(), 'z': np.random.rand(), 'size': 5, 'degree': 2} for i in range(10)]
        edges = [{'source': i, 'target': (i+1)%10} for i in range(10)]
        graph = {'nodes': nodes, 'edges': edges}
        json_out = cg.network_graph_visualization(graph)
        print("Network graph JSON generated successfully.")
    except Exception as e:
        print(f"Error: {e}")
