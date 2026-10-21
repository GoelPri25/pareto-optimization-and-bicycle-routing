import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import random
from deap import base, creator, tools, algorithms
from functools import partial
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ======================
# Configuration Parameters
# ======================
class Config:
    # Network parameters
    NUM_NODES = 20             # Total number of nodes
    START_NODE = 0              # Starting node
    TARGET_COUNT = 8            # Number of target nodes to visit
    
    # Algorithm parameters
    POPULATION_SIZE = 200       # Population size
    GENERATIONS = 100            # Number of generations
    CX_PROB = 0.7               # Crossover probability
    MUT_PROB = 0.2              # Mutation probability
    
    # Comfort handling method (options: 'objective', 'constraint', 'both')
    COMFORT_HANDLING = 'both'
    
    # Constraint parameters
    MIN_AVG_COMFORT = 3.0       # Minimum average comfort requirement
    MIN_EDGE_COMFORT = 2        # Minimum comfort per edge requirement
    COMFORT_PENALTY = 10000     # Penalty value for constraint violation

# ======================
# Data Generation
# ======================
def generate_data():
    """Generate distance matrix and comfort matrix"""
    np.random.seed(42)
    random.seed(42)
    
    # Generate distance matrix (random integers between 10 and 100)
    dist_matrix = np.random.randint(10, 100, size=(Config.NUM_NODES, Config.NUM_NODES))
    
    # Generate comfort matrix (random integers between 1 and 5)
    comfort_matrix = np.random.randint(1, 6, size=(Config.NUM_NODES, Config.NUM_NODES))
    
    # Make matrices symmetric and set diagonal to 0
    for i in range(Config.NUM_NODES):
        for j in range(Config.NUM_NODES):
            if i == j:
                dist_matrix[i][j] = 0
                comfort_matrix[i][j] = 0
            else:
                dist_matrix[i][j] = dist_matrix[j][i]
                comfort_matrix[i][j] = comfort_matrix[j][i]
    
    # Generate target nodes (excluding the start node)
    all_nodes = list(range(Config.NUM_NODES))
    all_nodes.remove(Config.START_NODE)
    target_nodes = random.sample(all_nodes, Config.TARGET_COUNT)
    
    return dist_matrix, comfort_matrix, target_nodes

# ======================
# Evaluation Function
# ======================
def evaluate_route(individual, dist_matrix, comfort_matrix, target_nodes, start_node):
    """Evaluate the fitness of a route"""
    # Map indices to actual nodes
    actual_route = [start_node] + [target_nodes[i] for i in individual] + [start_node]
    
    total_distance = 0
    total_comfort = 0
    min_comfort = float('inf')
    constraint_violation = 0
    
    # Calculate route metrics
    for i in range(len(actual_route) - 1):
        from_node = actual_route[i]
        to_node = actual_route[i+1]
        
        # Accumulate distance and comfort
        total_distance += dist_matrix[from_node][to_node]
        total_comfort += comfort_matrix[from_node][to_node]
        
        # Record minimum comfort
        if comfort_matrix[from_node][to_node] < min_comfort:
            min_comfort = comfort_matrix[from_node][to_node]
        
        # Check per-edge comfort constraint
        if Config.COMFORT_HANDLING in ['constraint', 'both']:
            if comfort_matrix[from_node][to_node] < Config.MIN_EDGE_COMFORT:
                constraint_violation += 1
    
    # Calculate average comfort
    avg_comfort = total_comfort / (len(actual_route) - 1)
    
    # Check average comfort constraint
    if Config.COMFORT_HANDLING in ['constraint', 'both']:
        if avg_comfort < Config.MIN_AVG_COMFORT:
            constraint_violation += 1
    
    # Apply penalty if constraints are violated
    penalty = Config.COMFORT_PENALTY * constraint_violation
    
    # Return fitness values based on handling method
    if Config.COMFORT_HANDLING == 'objective':
        return total_distance, avg_comfort
    elif Config.COMFORT_HANDLING == 'constraint':
        return (total_distance + penalty,)
    else:  # 'both'
        return total_distance + penalty, avg_comfort

# ======================
# Algorithm Setup
# ======================
def setup_algorithm(dist_matrix, comfort_matrix, target_nodes, start_node):
    """Configure genetic algorithm"""
    
    # Determine fitness weights based on comfort handling method
    if Config.COMFORT_HANDLING == 'objective':
        weights = (-1.0, 1.0)  # Minimize distance, maximize comfort
    elif Config.COMFORT_HANDLING == 'constraint':
        weights = (-1.0,)      # Minimize distance (with penalty)
    else:  # 'both'
        weights = (-1.0, 1.0)  # Minimize distance (with penalty), maximize comfort
    
    # Create fitness and individual classes
    creator.create("Fitness", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.Fitness)
    
    # Initialize toolbox
    toolbox = base.Toolbox()
    
    # Register individual and population creation functions
    toolbox.register("indices", random.sample, range(len(target_nodes)), len(target_nodes))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register evaluation function (fix parameters using partial)
    toolbox.register("evaluate", partial(evaluate_route, 
                                        dist_matrix=dist_matrix,
                                        comfort_matrix=comfort_matrix,
                                        target_nodes=target_nodes,
                                        start_node=start_node))
    
    # Register genetic operators
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
   
    # Choose selection operator based on number of objectives
    if len(weights) > 1:
        toolbox.register("select", tools.selNSGA2)
    else:
        toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox


def create_streamlined_visualization(hof, dist_matrix, comfort_matrix, target_nodes, start_node):
    """Create visualization of the route planning optimization"""    
    # Set modern style
    plt.style.use('dark_background')
    sns.set_palette("husl")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[2, 1], width_ratios=[2, 1])
    
    # Generate better node positions
    np.random.seed(42)
    angles = np.linspace(0, 2*np.pi, Config.NUM_NODES, endpoint=False)
    radius_base = 50
    node_positions = []
    
    for i in range(Config.NUM_NODES):
        radius = radius_base + np.random.normal(0, 8)
        angle = angles[i] + np.random.normal(0, 0.15)
        x = radius * np.cos(angle) + 50
        y = radius * np.sin(angle) + 50
        node_positions.append([x, y])
    
    node_positions = np.array(node_positions)
    
    # Get best route
    if len(hof[0].fitness.weights) > 1:
        distances = [ind.fitness.values[0] for ind in hof]
        comforts = [ind.fitness.values[1] for ind in hof]
        normalized_dist = (np.array(distances) - min(distances)) / (max(distances) - min(distances))
        normalized_comfort = (np.array(comforts) - min(comforts)) / (max(comforts) - min(comforts))
        scores = np.sqrt(normalized_dist**2 + (1 - normalized_comfort)**2)
        best_idx = np.argmin(scores)
        best_individual = hof[best_idx]
    else:
        best_individual = hof[0]
    
    actual_route = [start_node] + [target_nodes[i] for i in best_individual] + [start_node]
    
    #Main Network Visualization (Optimized Route Network)
    ax1 = fig.add_subplot(gs[:, 0])
    
    # Draw all possible connections with low opacity
    for i in range(Config.NUM_NODES):
        for j in range(i+1, Config.NUM_NODES):
            ax1.plot([node_positions[i, 0], node_positions[j, 0]], 
                    [node_positions[i, 1], node_positions[j, 1]], 
                    'gray', alpha=0.1, linewidth=0.5)
    
    for i in range(len(actual_route) - 1):
        from_node = actual_route[i]
        to_node = actual_route[i+1]
        comfort = comfort_matrix[from_node, to_node]
        distance = dist_matrix[from_node, to_node]
        
        colors = ['#ff4444', '#ff8844', '#ffaa44', '#44ff44', '#44ff88']
        color = colors[min(comfort-1, 4)]
        width = 1 + (comfort * 0.5)
        
        ax1.plot([node_positions[from_node, 0], node_positions[to_node, 0]],
                [node_positions[from_node, 1], node_positions[to_node, 1]],
                color=color, linewidth=width, alpha=0.8, zorder=2)
        
        # Add glowing effect
        ax1.plot([node_positions[from_node, 0], node_positions[to_node, 0]],
                [node_positions[from_node, 1], node_positions[to_node, 1]],
                color=color, linewidth=width*2, alpha=0.2, zorder=1)
        
        # Add distance labels
        mid_x = (node_positions[from_node, 0] + node_positions[to_node, 0]) / 2
        mid_y = (node_positions[from_node, 1] + node_positions[to_node, 1]) / 2
        
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7, edgecolor=color)
        ax1.text(mid_x, mid_y, f"{distance}", fontsize=9, ha='center', va='center',
                bbox=bbox_props, color='white', weight='bold')

    # Nodes
    for i in range(Config.NUM_NODES):
        if i == start_node:
            ax1.scatter(node_positions[i, 0], node_positions[i, 1], 
                       s=400, c='lime', marker='*', edgecolors='white', linewidths=2, zorder=3)
            ax1.text(node_positions[i, 0], node_positions[i, 1]-8, f'START\n{i}', 
                    ha='center', va='top', fontsize=10, weight='bold', color='lime')
        elif i in target_nodes:
            ax1.scatter(node_positions[i, 0], node_positions[i, 1], 
                       s=250, c='cyan', marker='s', edgecolors='white', linewidths=2, zorder=3)
            ax1.text(node_positions[i, 0], node_positions[i, 1]-6, str(i), 
                    ha='center', va='top', fontsize=9, weight='bold', color='cyan')
        else:
            ax1.scatter(node_positions[i, 0], node_positions[i, 1], 
                       s=100, c='gray', marker='o', edgecolors='white', linewidths=1, zorder=3)
            ax1.text(node_positions[i, 0], node_positions[i, 1]-4, str(i), 
                    ha='center', va='top', fontsize=8, color='gray')
    handles, labels = plt.gca().get_legend_handles_labels()
    
    legend_elements = [
        plt.Line2D([0], [0], color='#44ff44', lw=2, label='Comfort >= 4 (High)'),
        plt.Line2D([0], [0], color='#ffaa44', lw=2, label='Comfort >= 2.5 (Mid)'),
        plt.Line2D([0], [0], color='#ff4444', lw=2, label='Comfort < 2.5 (Low)')
    ]
    
    all_handles = handles + legend_elements
    all_labels = labels + [le.get_label() for le in legend_elements]
    plt.legend(all_handles, all_labels, fontsize='small', loc='best')
    
    
    ax1.set_title('Optimized Route Network', fontsize=16, weight='bold', color='white', pad=20)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#1a1a1a')
    
    # Pareto Front
    if len(hof[0].fitness.weights) > 1:
        ax2 = fig.add_subplot(gs[0, 1])
        
        distances = [ind.fitness.values[0] for ind in hof]
        comforts = [ind.fitness.values[1] for ind in hof]
        
        scatter = ax2.scatter(distances, comforts, c=comforts, cmap='viridis', 
                             s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
        
        # Highlight best compromise
        ax2.scatter(distances[best_idx], comforts[best_idx], 
                   s=200, c='red', marker='*', edgecolors='white', linewidths=2, zorder=5)
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Average Comfort', color='white')
        
        ax2.set_xlabel('Total Distance', fontsize=12, color='white')
        ax2.set_ylabel('Average Comfort', fontsize=12, color='white')
        ax2.set_title('Pareto Front Analysis', fontsize=14, weight='bold', color='white')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#1a1a1a')
    
    # Summary Statistics 
    ax3 = fig.add_subplot(gs[1, 1])
    
    route = [start_node] + [target_nodes[i] for i in best_individual] + [start_node]
    total_distance = sum(dist_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    avg_comfort = sum(comfort_matrix[route[i], route[i+1]] for i in range(len(route)-1)) / (len(route)-1)
    min_comfort = min(comfort_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    max_comfort = max(comfort_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    stats_text = f"""
    OPTIMAL ROUTE STATISTICS
    
    Route: {' → '.join(map(str, route))}
    
    Total Distance: {total_distance}
    Average Comfort: {avg_comfort:.2f}
    Min Comfort: {min_comfort}
    Max Comfort: {max_comfort}
    
    Nodes Visited: {len(route)-1}
    Population Size: {Config.POPULATION_SIZE}
    Generations: {Config.GENERATIONS}
    """
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', color='white', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#2a2a2a', alpha=0.8))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Optimal Route Statistics', fontsize=14, weight='bold', color='white')
    
    plt.suptitle('Route Planning Optimization', 
                fontsize=20, weight='bold', color='white', y=0.98)
    
    plt.tight_layout()
    plt.savefig("streamlined_route_analysis.png", dpi=300, bbox_inches='tight', 
                facecolor='#1a1a1a', edgecolor='none')
    plt.show()


def visualize_results(hof, dist_matrix, comfort_matrix, target_nodes, start_node):
    """Streamlined visualization of the best route, Pareto front and route statistics"""
    
    print("\nCreating Streamlined Visualizations...")
    
    try:
        create_streamlined_visualization(hof, dist_matrix, comfort_matrix, 
                                       target_nodes, start_node)
        print("Streamlined visualization created successfully!")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print_enhanced_route_details(hof, dist_matrix, comfort_matrix, target_nodes, start_node)

def print_enhanced_route_details(hof, dist_matrix, comfort_matrix, target_nodes, start_node):
    """Print detailed route analysis with enhanced formatting"""
    
    # Get best individual
    if len(hof[0].fitness.weights) > 1:
        distances = [ind.fitness.values[0] for ind in hof]
        comforts = [ind.fitness.values[1] for ind in hof]
        normalized_dist = (np.array(distances) - min(distances)) / (max(distances) - min(distances))
        normalized_comfort = (np.array(comforts) - min(comforts)) / (max(comforts) - min(comforts))
        scores = np.sqrt(normalized_dist**2 + (1 - normalized_comfort)**2)
        best_idx = np.argmin(scores)
        best_individual = hof[best_idx]
    else:
        best_individual = hof[0]
    
    actual_route = [start_node] + [target_nodes[i] for i in best_individual] + [start_node]
    
    print("\n" + "="*80)
    print("OPTIMAL ROUTE ANALYSIS")
    print("="*80)
    
    # Route overview
    print(f"Route Path: {' → '.join(map(str, actual_route))}")
    print(f"Target Nodes: {sorted(target_nodes)}")
    print(f"Starting Node: {start_node}")
    
    # Summary statistics
    total_distance = sum(dist_matrix[actual_route[i], actual_route[i+1]] for i in range(len(actual_route)-1))
    avg_comfort = sum(comfort_matrix[actual_route[i], actual_route[i+1]] for i in range(len(actual_route)-1)) / (len(actual_route) - 1)
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total Distance:     {total_distance}")
    print(f"Average Comfort:    {avg_comfort:.2f}")
    print(f"Min Comfort:        {min(comfort_matrix[actual_route[i], actual_route[i+1]] for i in range(len(actual_route)-1))}")
    print(f"Max Comfort:        {max(comfort_matrix[actual_route[i], actual_route[i+1]] for i in range(len(actual_route)-1))}")
    print(f"Segments Count:     {len(actual_route)-1}")
    
    print("="*80)

def main():

    dist_matrix, comfort_matrix, target_nodes = generate_data()

    
    toolbox = setup_algorithm(dist_matrix, comfort_matrix, target_nodes, Config.START_NODE)
    
    population = toolbox.population(n=Config.POPULATION_SIZE)
    
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    
    if len(toolbox.individual().fitness.weights) > 1:
        stats.register("min_dist", lambda x: min(val[0] for val in x))
        stats.register("max_comfort", lambda x: max(val[1] for val in x))
    else:
        stats.register("min", np.min)
        stats.register("avg", np.mean)

    
    print("\n🏃‍♂️ Running optimization...")
    
    # Run genetic algorithm with progress tracking
    result_pop, logbook = algorithms.eaMuPlusLambda(
        population, toolbox, 
        mu=Config.POPULATION_SIZE, lambda_=Config.POPULATION_SIZE,
        cxpb=Config.CX_PROB, mutpb=Config.MUT_PROB,
        ngen=Config.GENERATIONS, stats=stats,
        halloffame=hof, verbose=True
    )
    
    
    visualize_results(hof, dist_matrix, comfort_matrix, target_nodes, Config.START_NODE)

if __name__ == "__main__":
    main()