import numpy as np
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from functools import partial
import seaborn as sns
# ======================
# Configuration Parameters
# ======================
class Config:
    # Network parameters
    NUM_NODES =20             # Total number of nodes
    START_NODE = 0              # Starting node
    TARGET_COUNT = 8            # Number of target nodes to visit
    
    # Algorithm parameters
    POPULATION_SIZE = 200       # Population size
    GENERATIONS = 100            # Number of generations
    CX_PROB = 0.7               # Crossover probability
    MUT_PROB = 0.2              # Mutation probability
    
    # Comfort handling method (options: 'desirability', 'objective', 'constraint', 'both')
    COMFORT_HANDLING = 'both'
    # Constraint parameters
    MIN_AVG_COMFORT = 12.0       # Minimum average comfort requirement
    MIN_EDGE_COMFORT = 8        # Minimum comfort per edge requirement
    COMFORT_PENALTY = 10000     # Penalty value for constraint violation

# ======================
# Data Generation
# ======================
def generate_data():
    np.random.seed(42)
    random.seed(42)

    dist_matrix = np.random.randint(10, 100, size=(Config.NUM_NODES, Config.NUM_NODES))

    # Raw rating matrices for each dimension (1~5)
    scenic = np.random.randint(1, 6, size=(Config.NUM_NODES, Config.NUM_NODES))
    roughness = np.random.randint(1, 6, size=(Config.NUM_NODES, Config.NUM_NODES))
    safety = np.random.randint(1, 6, size=(Config.NUM_NODES, Config.NUM_NODES))
    slope = np.random.randint(1, 6, size=(Config.NUM_NODES, Config.NUM_NODES))

    # Construct comfort matrix (invert safety and slope)
    comfort_matrix = (
        scenic +
        roughness +
        (6 - safety) +
        (6 - slope)
    )

    # Enforce symmetry and zero diagonal
    for i in range(Config.NUM_NODES):
        for j in range(Config.NUM_NODES):
            if i == j:
                dist_matrix[i][j] = 0
                comfort_matrix[i][j] = 0
            else:
                dist_matrix[i][j] = dist_matrix[j][i]
                comfort_matrix[i][j] = comfort_matrix[j][i]

    all_nodes = list(range(Config.NUM_NODES))
    all_nodes.remove(Config.START_NODE)
    target_nodes = random.sample(all_nodes, Config.TARGET_COUNT)

    return dist_matrix, comfort_matrix, target_nodes


# ======================
# Evaluation Function
# ======================
def evaluate_route(individual, dist_matrix, comfort_matrix, target_nodes, start_node):
    """Evaluate the fitness of a route"""
    actual_route = [start_node] + [target_nodes[i] for i in individual] + [start_node]
    
    total_distance = 0
    total_comfort = 0
    constraint_violation = 0
    
    for i in range(len(actual_route) - 1):
        from_node = actual_route[i]
        to_node = actual_route[i+1]
        
        dist = dist_matrix[from_node][to_node]
        comfort = comfort_matrix[from_node][to_node]
        
        total_distance += dist
        total_comfort += comfort

        # Constraints check
        if Config.COMFORT_HANDLING in ['constraint', 'both']:
            if comfort < Config.MIN_EDGE_COMFORT:
                constraint_violation += 1
    
    avg_comfort = total_comfort / (len(actual_route) - 1)
    
    if Config.COMFORT_HANDLING in ['constraint', 'both']:
        if avg_comfort < Config.MIN_AVG_COMFORT:
            constraint_violation += 1

    penalty = Config.COMFORT_PENALTY * constraint_violation

    # --- Desirability Handling ---
    if Config.COMFORT_HANDLING == 'desirability':
        # Normalize using some reasonable fixed bounds
        d_min, d_max = 200, 1000
        c_min, c_max = 1, 20

        desirability_distance = max(0.0, min(1.0, ((d_max - total_distance) / (d_max - d_min)) ** 2))
        desirability_comfort = max(0.0, min(1.0, ((avg_comfort - c_min) / (c_max - c_min)) ** 2))

        combined_desirability = (desirability_distance * desirability_comfort) ** 0.5
        return (1 - combined_desirability, )  # we minimize in DEAP
    # --- End desirability ---

    if Config.COMFORT_HANDLING == 'objective':
        return total_distance, avg_comfort
    elif Config.COMFORT_HANDLING == 'desirability':
        return (1 - combined_desirability, )
    elif Config.COMFORT_HANDLING == 'constraint':
        return (total_distance + penalty, )
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
    elif Config.COMFORT_HANDLING == 'desirability':
        weights = (-1.0,)
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

# ======================
# Result Visualization
# ======================


def plot_black_nodes_only(node_positions, target_nodes, start_node):
    """Plot black (non-target, non-start) nodes, with target and start nodes highlighted separately."""
    black_nodes = [i for i in range(Config.NUM_NODES) if i not in target_nodes and i != start_node]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.style.use('default')
    
    # Plot black nodes
    for i in black_nodes:
        ax.scatter(node_positions[i, 0], node_positions[i, 1], 
                   s=100, c='gray', marker='o', edgecolors='white', linewidths=1, zorder=3, label="Other Nodes")
        ax.text(node_positions[i, 0], node_positions[i, 1] - 4, str(i), 
                ha='center', va='top', fontsize=8, color='gray')
    
    # Plot target nodes
    for i in target_nodes:
        ax.scatter(node_positions[i, 0], node_positions[i, 1], 
                   s=250, c='cyan', marker='s', edgecolors='white', linewidths=2, zorder=4, label='Target Nodes')
        ax.text(node_positions[i, 0], node_positions[i, 1] - 6, str(i), 
                ha='center', va='top', fontsize=9, weight='bold', color='cyan')
    
    # Plot start node
    ax.scatter(node_positions[start_node, 0], node_positions[start_node, 1], 
               s=400, c='pink', marker='s', edgecolors='white', linewidths=2, zorder=5, label='Start Node')
    ax.text(node_positions[start_node, 0], node_positions[start_node, 1] - 8, f'START\n{start_node}', 
            ha='center', va='top', fontsize=10, weight='bold', color='lime')

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict()
    
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    # Unpack back to lists
    unique_labels = list(unique.keys())
    unique_handles = list(unique.values())
    
    
    
    ax.set_title(f'Bicycle locations: \n Total Nodes:{len(black_nodes)+ len(target_nodes)+ 1}, Target Nodes: {len(target_nodes)}', fontsize=14, weight='bold', color='black')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.5)
    ax.set_facecolor("#f6f3f3")
    plt.legend(unique_handles, unique_labels, fontsize='small', loc='lower left')
    plt.tight_layout()
    plt.savefig("black_target_start_nodes.png", dpi=300, facecolor='#1a1a1a')
    plt.show()

def create_streamlined_visualization(hof, dist_matrix, comfort_matrix, target_nodes, start_node):
    """Create visualization of the route planning optimization"""    
    # Set modern style
    plt.style.use('default')
    #plt.style.use('default')

    #sns.set_palette("husl")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[2, 1], width_ratios=[2, 1])
    
    # Generate better node positions
    
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
            
    
    route = [start_node] + [target_nodes[i] for i in best_individual] + [start_node]
    
    total_distance = sum(dist_matrix[actual_route[i], actual_route[i+1]] for i in range(len(actual_route)-1))
    avg_comfort = sum(comfort_matrix[actual_route[i], actual_route[i+1]] for i in range(len(actual_route)-1)) / (len(actual_route)-1)
    print(avg_comfort)
    plt.title(f"Best Route (Distance: {total_distance:.1f}, Avg Comfort: {avg_comfort:.2f})", fontsize=16, weight='bold', color='black')
    
    min_comfort = min(comfort_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    max_comfort = max(comfort_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    for i in range(len(actual_route) - 1):
        from_node = actual_route[i]
        to_node = actual_route[i+1]
        comfort = comfort_matrix[from_node, to_node]
        distance = dist_matrix[from_node, to_node]
        
        # colors = ['#ff4444', '#ff8844', '#ffaa44', '#44ff44', '#44ff88']
        
        if comfort >= 16:
            color = '#44ff44'  # High comfort
        elif comfort >= 12:
            color = '#ffaa44'
        else:
            color = '#ff4444'
        
        width = 1 + 1.5*(abs((comfort -5)) / 4)  # Scale width based on comfort
        print(f"Comfort: {comfort}, Distance: {distance}")
   
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
        
        seg_dx = node_positions[to_node, 0] - node_positions[from_node, 0]
        seg_dy = node_positions[to_node, 1] - node_positions[from_node, 1]
        segment_len = np.sqrt(seg_dx**2 + seg_dy**2)

        norm_seg_dx, norm_seg_dy = 0, 0
        if segment_len > 1e-6: # Avoid division by zero for very short/zero length segments
            norm_seg_dx = seg_dx / segment_len
            norm_seg_dy = seg_dy / segment_len
        
        trace_label_offset = 5  # Distance from the start of the segment along its path
                                 # Adjust this value based on node size and desired placement

        # Only draw trace label if segment is long enough and not a zero-length segment
        if segment_len > trace_label_offset :
            trace_x = node_positions[from_node, 0] + norm_seg_dx * trace_label_offset
            trace_y = node_positions[from_node, 1] + norm_seg_dy * trace_label_offset
            
            bbox_props_trace = dict(boxstyle="circle,pad=0.1", facecolor='white', alpha=0.6, edgecolor='black')
            ax1.text(trace_x, trace_y, f"#{i+1}", fontsize=7, ha='center', va='center',
                     color='black',  bbox=bbox_props_trace, zorder=5) # Ensure it's on top
            
    # Nodes
    for i in range(Config.NUM_NODES):
        if i == start_node:
            ax1.scatter(node_positions[i, 0], node_positions[i, 1], 
                       s=400, c='pink', marker='s', edgecolors='white', linewidths=2, zorder=3, label = "Start Node")
            ax1.text(node_positions[i, 0], node_positions[i, 1]-8, f'START\n{i}', 
                    ha='center', va='top', fontsize=10, weight='bold', color='lime')
        elif i in target_nodes:
            ax1.scatter(node_positions[i, 0], node_positions[i, 1], 
                       s=250, c='cyan', marker='s', edgecolors='white', linewidths=2, zorder=3, label = "Target Nodes")
            ax1.text(node_positions[i, 0], node_positions[i, 1]-6, str(i), 
                    ha='center', va='top', fontsize=9, weight='bold', color='cyan')
        else:
            ax1.scatter(node_positions[i, 0], node_positions[i, 1], 
                       s=100, c='gray', marker='o', edgecolors='white', linewidths=1, zorder=3, label = "Other Nodes")
            ax1.text(node_positions[i, 0], node_positions[i, 1]-4, str(i), 
                    ha='center', va='top', fontsize=8, color='gray')
    handles, labels = plt.gca().get_legend_handles_labels()
    
    legend_elements = [
        plt.Line2D([0], [0], color='#44ff44', lw=2, label='Comfort >= 16 (High)'),
        plt.Line2D([0], [0], color='#ffaa44', lw=2, label='Comfort >= 12 (Mid)'),
        plt.Line2D([0], [0], color='#ff4444', lw=2, label='Comfort < 12 (Low)')
    ]
    
    all_handles = handles + legend_elements
    
    
    all_labels = labels + [le.get_label() for le in legend_elements]
    
    unique = dict()
    for h, l in zip(all_handles, all_labels):
        if l not in unique:
            unique[l] = h

    # Unpack back to lists
    unique_labels = list(unique.keys())
    unique_handles = list(unique.values())
    
    plt.legend(unique_handles, unique_labels, fontsize='small', loc='best')
    
    #ax1.set_title(f"Best Route (Distance: {total_distance:.1f}, Avg Comfort: {avg_comfort:.2f})", fontsize=16, weight='bold', color='Black', pad=20)
    
    
    # plt.xlabel("Total Distance(in m)")
    # plt.ylabel("Average Comfort")
    #ax1.set_title('Optimized Route Network', fontsize=16, weight='bold', color='white', pad=20)
   
    #plt.suptitle(f"Best Route (Distance: {total_distance:.1f}, Avg Comfort: {avg_comfort:.2f})", fontsize=20, weight='bold', color='white', y=0.98)    
    

    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    #ax1.set_facecolor('#1a1a1a')
    ax1.set_facecolor("#f6f3f3")
    
    # Pareto Front
    if len(hof[0].fitness.weights) > 1:
        ax2 = fig.add_subplot(gs[0, 1])
        
        distances = [ind.fitness.values[0] for ind in hof]
        comforts = [ind.fitness.values[1] for ind in hof]
        
        scatter = ax2.scatter(distances, comforts, c=comforts, cmap='viridis', 
                             s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
        
        # Highlight best compromise
        ax2.scatter(distances[best_idx], comforts[best_idx], 
                   s=200, c='red', marker='*', edgecolors='Red', linewidths=2, zorder=5, label='Best Compromise')
        
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Average Comfort', color='white')
        
        ax2.set_xlabel('Total Distance(in m)', fontsize=12, color='black')
        ax2.set_ylabel('Average Comfort', fontsize=12, color='black')
        ax2.set_title('Pareto Front Analysis', fontsize=14, weight='bold', color='black')
        ax2.grid(True, alpha=0.3)
        
        ax2.set_facecolor("#f6f3f3")
        plt.legend(fontsize='small', loc='best', facecolor='white', edgecolor='black')
    # Summary Statistics 
    ax3 = fig.add_subplot(gs[1, 1])
    
    
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
            verticalalignment='top', color='black', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Optimal Route Statistics', fontsize=14, weight='bold', color='black')
    
    plt.suptitle('Route Planning Optimization', 
                fontsize=20, weight='bold', color='black', y=0.98)
    
    plt.tight_layout()
    plt.savefig("streamlined_route_analysis.png", dpi=300, bbox_inches='tight', 
                facecolor='#1a1a1a', edgecolor='none')
    
    plt.show()
    return node_positions, target_nodes, start_node

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


def visualize_results(hof, dist_matrix, comfort_matrix, target_nodes, start_node):
    """Streamlined visualization of the best route, Pareto front and route statistics"""
    
    print("\nCreating Streamlined Visualizations...")
    
    #try:
    node_positions, target_nodes, start_nodes = create_streamlined_visualization(hof, dist_matrix, comfort_matrix, 
                                    target_nodes, start_node)
    print("Streamlined visualization created successfully!")

    plot_black_nodes_only(node_positions, target_nodes, start_node)
    print_enhanced_route_details(hof, dist_matrix, comfort_matrix, target_nodes, start_nodes)


# ======================
# Main Function
# ======================
def main():
    # Generate data
    dist_matrix, comfort_matrix, target_nodes = generate_data()
    
    print("="*50)
    print("Route Planning Parameter Configuration:")
    print(f"Total Nodes: {Config.NUM_NODES}")
    print(f"Start Node: {Config.START_NODE}")
    print(f"Target Nodes: {target_nodes}")
    print(f"Comfort Handling Method: {Config.COMFORT_HANDLING}")
    if Config.COMFORT_HANDLING in ['constraint', 'both']:
        print(f"Minimum Edge Comfort: {Config.MIN_EDGE_COMFORT}")
        print(f"Minimum Average Comfort: {Config.MIN_AVG_COMFORT}")
    print("="*50)
    
    # Setup algorithm
    toolbox = setup_algorithm(dist_matrix, comfort_matrix, target_nodes, Config.START_NODE)
    
    # Create initial population
    population = toolbox.population(n=Config.POPULATION_SIZE)
    
    # Setup Hall of Fame to store best individuals
    #hof = tools.HallOfFame(10)
    hof = tools.ParetoFront()
    
    # Setup statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    if len(toolbox.individual().fitness.weights) > 1:
        stats.register("min_dist", lambda x: min(val[0] for val in x))
        stats.register("max_comfort", lambda x: max(val[1] for val in x))
    else:
        stats.register("min", np.min)
    
    # Run genetic algorithm
    # algorithms.eaSimple(
    #     population, toolbox, 
    #     cxpb=Config.CX_PROB, mutpb=Config.MUT_PROB,
    #     ngen=Config.GENERATIONS, stats=stats,
    #     halloffame=hof, verbose=True
    # )
    
    algorithms.eaMuPlusLambda(
        population, toolbox, 
        mu=Config.POPULATION_SIZE, lambda_=Config.GENERATIONS,
        cxpb=Config.CX_PROB, mutpb=Config.MUT_PROB,
        ngen=Config.GENERATIONS, stats=stats,
        halloffame=hof, verbose=True
    )
    
    # Visualize results
    visualize_results(hof, dist_matrix, comfort_matrix, target_nodes, Config.START_NODE)

if __name__ == "__main__":
    main()