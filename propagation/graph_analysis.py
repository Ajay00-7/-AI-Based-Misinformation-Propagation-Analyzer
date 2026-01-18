"""
Network Graph Analysis for Misinformation Propagation
Creates and analyzes social network graphs to understand spread patterns
"""

import networkx as nx
import numpy as np
from collections import defaultdict

class GraphAnalysis:
    def __init__(self, num_users=50, connection_prob=0.1):
        """
        Initialize graph analysis
        
        Args:
            num_users: Number of users in the network
            connection_prob: Probability of connection between users
        """
        self.num_users = num_users
        self.connection_prob = connection_prob
        self.graph = None
        
    def create_network(self, network_type='small_world'):
        """
        Create social network graph
        
        Args:
            network_type: Type of network ('small_world', 'scale_free', 'random')
            
        Returns:
            NetworkX graph object
        """
        if network_type == 'small_world':
            # Watts-Strogatz small-world graph (realistic social network)
            self.graph = nx.watts_strogatz_graph(self.num_users, k=6, p=0.3)
        elif network_type == 'scale_free':
            # Barabási-Albert scale-free graph (has influencers)
            self.graph = nx.barabasi_albert_graph(self.num_users, m=3)
        else:
            # Random Erdős-Rényi graph
            self.graph = nx.erdos_renyi_graph(self.num_users, self.connection_prob)
        
        return self.graph
    
    def identify_influencers(self, top_n=10):
        """
        Identify influential nodes in the network
        
        Args:
            top_n: Number of top influencers to return
            
        Returns:
            dict: Top influencers with their metrics
        """
        if self.graph is None:
            self.create_network()
        
        # Calculate centrality metrics
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph)
        
        # Combine metrics (weighted average)
        influence_score = {}
        for node in self.graph.nodes():
            influence_score[node] = (
                0.5 * degree_centrality[node] +
                0.3 * betweenness_centrality[node] +
                0.2 * closeness_centrality[node]
            )
        
        # Get top influencers
        top_influencers = sorted(
            influence_score.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        influencer_data = []
        for node, score in top_influencers:
            influencer_data.append({
                'id': int(node),
                'influence_score': float(score),
                'degree': int(self.graph.degree(node)),
                'betweenness': float(betweenness_centrality[node]),
                'closeness': float(closeness_centrality[node])
            })
        
        return influencer_data
    
    def simulate_diffusion(self, seed_nodes=None, steps=20, infection_prob=0.3):
        """
        Simulate information diffusion through network
        
        Args:
            seed_nodes: Initial nodes that have the information
            steps: Number of time steps to simulate
            infection_prob: Probability of transmission per edge
            
        Returns:
            dict: Diffusion statistics over time
        """
        if self.graph is None:
            self.create_network()
        
        # Initialize
        if seed_nodes is None:
            # Start with top influencers
            influencers = self.identify_influencers(top_n=5)
            seed_nodes = [inf['id'] for inf in influencers]
        
        infected = set(seed_nodes)
        infection_timeline = [len(infected)]
        
        # Simulate spread
        for step in range(steps):
            newly_infected = set()
            
            for node in infected:
                neighbors = list(self.graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in infected:
                        # Probabilistic infection
                        if np.random.random() < infection_prob:
                            newly_infected.add(neighbor)
            
            infected.update(newly_infected)
            infection_timeline.append(len(infected))
            
            # Stop if no new infections
            if len(newly_infected) == 0:
                break
        
        return {
            'timeline': infection_timeline,
            'final_infected': len(infected),
            'reach_percentage': (len(infected) / self.num_users) * 100,
            'steps': len(infection_timeline) - 1
        }
    
    def get_network_stats(self):
        """
        Get network statistics
        
        Returns:
            dict: Network metrics
        """
        if self.graph is None:
            self.create_network()
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': float(np.mean([d for n, d in self.graph.degree()])),
            'density': float(nx.density(self.graph)),
            'clustering_coefficient': float(nx.average_clustering(self.graph)),
            'diameter': int(nx.diameter(self.graph)) if nx.is_connected(self.graph) else -1
        }
    
    def get_graph_layout_data(self):
        """
        Get graph data for visualization
        
        Returns:
            dict: Node and edge data for plotting
        """
        if self.graph is None:
            self.create_network()
        
        # Use spring layout for positioning
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Get influencers for node sizing
        influencers = self.identify_influencers(top_n=self.num_users)
        influence_dict = {inf['id']: inf['influence_score'] for inf in influencers}
        
        # Node data
        nodes = []
        for node in self.graph.nodes():
            nodes.append({
                'id': int(node),
                'x': float(pos[node][0]),
                'y': float(pos[node][1]),
                'size': float(influence_dict.get(node, 0.1)) * 20,  # Scale for visibility
                'degree': int(self.graph.degree(node))
            })
        
        # Edge data
        edges = []
        for edge in self.graph.edges():
            edges.append({
                'source': int(edge[0]),
                'target': int(edge[1])
            })
        
        return {
            'nodes': nodes,
            'edges': edges
        }


# For quick testing
if __name__ == "__main__":
    graph_analysis = GraphAnalysis(num_users=100)
    
    print("Creating social network...")
    graph_analysis.create_network('scale_free')
    
    print("\nNetwork Statistics:")
    stats = graph_analysis.get_network_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nTop Influencers:")
    influencers = graph_analysis.identify_influencers(top_n=5)
    for i, inf in enumerate(influencers, 1):
        print(f"{i}. User {inf['id']}: Score={inf['influence_score']:.3f}, Degree={inf['degree']}")
    
    print("\nSimulating Information Diffusion:")
    diffusion = graph_analysis.simulate_diffusion(steps=20, infection_prob=0.3)
    print(f"Final Reach: {diffusion['final_infected']} users ({diffusion['reach_percentage']:.1f}%)")
    print(f"Spread in {diffusion['steps']} steps")
