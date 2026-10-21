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
    COMFORT_HANDLING = 'both' # 'objective' or 'both' for Pareto, 'constraint' or 'desirability' for single-objective
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
    ) # Max comfort = 5+5+5+5 = 20, Min comfort = 1+1+1+1 = 4

    # Enforce symmetry and zero diagonal
    for i in range(Config.NUM_NODES):
        for j in range(Config.NUM_NODES):
            if i == j:
                dist_matrix[i][j] = 0
                comfort_matrix[i][j] = 0 # Comfort on a path to itself is 0, or could be max? For calculation, 0 is fine.
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
    
    # Check for empty or single-point route (should not happen with actual_route construction)
    if len(actual_route) <= 1:
        # Heavily penalize if route is trivial (e.g. just start_node -> start_node)
        # This case might occur if target_nodes is empty, but problem setup prevents this.
        if Config.COMFORT_HANDLING == 'objective' or Config.COMFORT_HANDLING == 'both':
            return float('inf'), 0 # Infinite distance, zero comfort
        elif Config.COMFORT_HANDLING == 'desirability':
            return (1.0, ) # Worst desirability
        else: # constraint
            return (float('inf'), )


    num_edges = len(actual_route) - 1

    for i in range(num_edges):
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
    
    avg_comfort = total_comfort / num_edges if num_edges > 0 else 0
    
    if Config.COMFORT_HANDLING in ['constraint', 'both']:
        if avg_comfort < Config.MIN_AVG_COMFORT:
            constraint_violation += 1

    penalty = Config.COMFORT_PENALTY * constraint_violation

    # --- Desirability Handling ---
    if Config.COMFORT_HANDLING == 'desirability':
        # Normalize using some reasonable fixed bounds
        # These bounds should ideally be derived from data or problem context
        # Example bounds: min distance for TARGET_COUNT+1 segments, max distance
        # min comfort rating, max comfort rating
        d_min_est = Config.TARGET_COUNT * 10 # Estimated min possible distance
        d_max_est = Config.TARGET_COUNT * 100 # Estimated max possible distance
        c_min_rating = 4 # Min possible comfort per edge
        c_max_rating = 20 # Max possible comfort per edge

        # Avoid division by zero if d_max_est == d_min_est or c_max_rating == c_min_rating
        desirability_distance = 0.0
        if d_max_est > d_min_est:
             desirability_distance = max(0.0, min(1.0, ((d_max_est - total_distance) / (d_max_est - d_min_est)) ** 2)) # Higher is better
        
        desirability_comfort = 0.0
        if c_max_rating > c_min_rating:
            desirability_comfort = max(0.0, min(1.0, ((avg_comfort - c_min_rating) / (c_max_rating - c_min_rating)) ** 2)) # Higher is better

        combined_desirability = (desirability_distance * desirability_comfort) ** 0.5
        return (1 - combined_desirability + penalty, ) # we minimize in DEAP, add penalty if constraints are also active
    # --- End desirability ---

    if Config.COMFORT_HANDLING == 'objective':
        return total_distance, -avg_comfort # Maximize avg_comfort by minimizing -avg_comfort
    elif Config.COMFORT_HANDLING == 'constraint':
        return (total_distance + penalty, )
    else:  # 'both'
        return total_distance + penalty, -avg_comfort # Maximize avg_comfort

# ======================
# Algorithm Setup
# ======================
def setup_algorithm(dist_matrix, comfort_matrix, target_nodes, start_node):
    """Configure genetic algorithm"""
    
    # Determine fitness weights based on comfort handling method
    if Config.COMFORT_HANDLING == 'objective':
        weights = (-1.0, -1.0)  # Minimize distance, Minimize -avg_comfort (maximize comfort)
    elif Config.COMFORT_HANDLING == 'constraint':
        weights = (-1.0,)      # Minimize distance (with penalty)
    elif Config.COMFORT_HANDLING == 'desirability':
        weights = (-1.0,)      # Minimize (1 - desirability)
    else:  # 'both'
        weights = (-1.0, -1.0)  # Minimize distance (with penalty), Minimize -avg_comfort (maximize comfort)
    
    creator.create("Fitness", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.Fitness)
    
    toolbox = base.Toolbox()
    
    toolbox.register("indices", random.sample, range(len(target_nodes)), len(target_nodes))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", partial(evaluate_route, 
                                        dist_matrix=dist_matrix,
                                        comfort_matrix=comfort_matrix,
                                        target_nodes=target_nodes,
                                        start_node=start_node))
    
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1) # indpb is prob of swapping each index
   
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
    plt.style.use('dark_background')
    
    for i in black_nodes:
        ax.scatter(node_positions[i, 0], node_positions[i, 1], 
                   s=100, c='gray', marker='o', edgecolors='white', linewidths=1, zorder=3)
        ax.text(node_positions[i, 0], node_positions[i, 1] - 4, str(i), 
                ha='center', va='top', fontsize=8, color='gray')
    
    for i in target_nodes:
        ax.scatter(node_positions[i, 0], node_positions[i, 1], 
                   s=250, c='cyan', marker='s', edgecolors='white', linewidths=2, zorder=4)
        ax.text(node_positions[i, 0], node_positions[i, 1] - 6, str(i), 
                ha='center', va='top', fontsize=9, weight='bold', color='cyan')
    
    ax.scatter(node_positions[start_node, 0], node_positions[start_node, 1], 
               s=400, c='lime', marker='*', edgecolors='white', linewidths=2, zorder=5)
    ax.text(node_positions[start_node, 0], node_positions[start_node, 1] - 8, f'START\n{start_node}', 
            ha='center', va='top', fontsize=10, weight='bold', color='lime')

    ax.set_title(f'Bicycle locations: \n Total Nodes {Config.NUM_NODES}, Target Nodes: {len(target_nodes)}', fontsize=14, weight='bold', color='white')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#1a1a1a')
    plt.tight_layout()
    plt.savefig("black_target_start_nodes.png", dpi=300, facecolor='#1a1a1a')
    plt.show()

def create_streamlined_visualization(hof, dist_matrix, comfort_matrix, target_nodes, start_node):
    """Create visualization of the route planning optimization"""    
    plt.style.use('dark_background')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[2, 1], width_ratios=[2, 1])
    
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
    
    best_individual = None
    if not hof: # Hall of Fame is empty
        print("Warning: Hall of Fame is empty. Cannot create visualization.")
        # Create a dummy plot or return
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.text(0.5, 0.5, "No solution found / HOF empty.", ha='center', va='center', color='red', fontsize=14)
        ax1.set_facecolor('#1a1a1a')
        plt.savefig("streamlined_route_analysis_error.png", dpi=300, facecolor='#1a1a1a')
        plt.show()
        return node_positions, target_nodes, start_node # Return generated positions anyway for plot_black_nodes_only


    if len(hof[0].fitness.weights) > 1: # Multi-objective
        # Select a "best" compromise from Pareto front
        # Using a simple Euclidean distance from ideal point (min_dist, max_comfort i.e. min -(-avg_comfort))
        distances = np.array([ind.fitness.values[0] for ind in hof])
        # Fitness stores -avg_comfort, so take negative to get actual avg_comfort
        comforts = np.array([-ind.fitness.values[1] for ind in hof]) 

        if not distances.size or not comforts.size : # Handle empty arrays after filtering
            print("Warning: No valid individuals in HOF after filtering for visualization.")
            best_individual = hof[0] # Fallback to the first one
        else:
            # Normalize: (val - min) / (max - min)
            # For distance, lower is better. For comfort, higher is better.
            # Ideal point: (min_dist, max_comfort)
            # We want to minimize sqrt( ((d - min_d)/(max_d-min_d))^2 + ((max_c - c)/(max_c-min_c))^2 )
            norm_dist = (distances - distances.min()) / (distances.max() - distances.min()) if distances.max() > distances.min() else np.zeros_like(distances)
            norm_comfort = (comforts.max() - comforts) / (comforts.max() - comforts.min()) if comforts.max() > comforts.min() else np.zeros_like(comforts)
            
            scores = np.sqrt(norm_dist**2 + norm_comfort**2)
            best_idx = np.argmin(scores)
            best_individual = hof[best_idx]
    else: # Single-objective
        best_individual = hof[0]
    
    actual_route = [start_node] + [target_nodes[i] for i in best_individual] + [start_node]
    
    ax1 = fig.add_subplot(gs[:, 0])
    
    for i in range(Config.NUM_NODES):
        for j in range(i+1, Config.NUM_NODES):
            ax1.plot([node_positions[i, 0], node_positions[j, 0]], 
                    [node_positions[i, 1], node_positions[j, 1]], 
                    'gray', alpha=0.1, linewidth=0.5)
            
    total_distance_val = sum(dist_matrix[actual_route[k], actual_route[k+1]] for k in range(len(actual_route)-1))
    avg_comfort_val = sum(comfort_matrix[actual_route[k], actual_route[k+1]] for k in range(len(actual_route)-1)) / (len(actual_route)-1)
    
    min_route_comfort = float('inf')
    max_route_comfort = float('-inf')
    if len(actual_route) > 1:
        min_route_comfort = min(comfort_matrix[actual_route[k], actual_route[k+1]] for k in range(len(actual_route)-1))
        max_route_comfort = max(comfort_matrix[actual_route[k], actual_route[k+1]] for k in range(len(actual_route)-1))
    else: # Should not happen with valid route
        min_route_comfort = 0
        max_route_comfort = 0


    for i in range(len(actual_route) - 1):
        from_node = actual_route[i]
        to_node = actual_route[i+1]
        comfort = comfort_matrix[from_node, to_node]
        distance_val = dist_matrix[from_node, to_node]
        
        if comfort >= 16: color = '#44ff44'  # High comfort
        elif comfort >= 12: color = '#ffaa44' # Mid comfort
        else: color = '#ff4444' # Low comfort
        
        # Scale width based on comfort (example scaling)
        # Assuming comfort range [4, 20]
        # Map comfort to width [1, 3] -> (comfort - 4) / (20 - 4) * (3-1) + 1
        width = 1 + ( (comfort - 4) / 16 ) * 2 if comfort >=4 else 1 # Clamp min width to 1
        width = max(1, min(width, 3)) # Ensure width is in [1, 3]

        # Plot main segment
        ax1.plot([node_positions[from_node, 0], node_positions[to_node, 0]],
                [node_positions[from_node, 1], node_positions[to_node, 1]],
                color=color, linewidth=width, alpha=0.8, zorder=2)
        
        # Plot glow effect
        ax1.plot([node_positions[from_node, 0], node_positions[to_node, 0]],
                [node_positions[from_node, 1], node_positions[to_node, 1]],
                color=color, linewidth=width*2, alpha=0.2, zorder=1)
        
        # Add distance labels
        mid_x = (node_positions[from_node, 0] + node_positions[to_node, 0]) / 2
        mid_y = (node_positions[from_node, 1] + node_positions[to_node, 1]) / 2
        
        bbox_props_dist = dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7, edgecolor=color)
        ax1.text(mid_x, mid_y, f"{distance_val}", fontsize=8, ha='center', va='center',
                bbox=bbox_props_dist, color='white', weight='bold', zorder=4)

        # ====================================================================
        # ADDED: Segment number marker for tracing
        # ====================================================================
        seg_dx = node_positions[to_node, 0] - node_positions[from_node, 0]
        seg_dy = node_positions[to_node, 1] - node_positions[from_node, 1]
        segment_len = np.sqrt(seg_dx**2 + seg_dy**2)

        norm_seg_dx, norm_seg_dy = 0, 0
        if segment_len > 1e-6: # Avoid division by zero for very short/zero length segments
            norm_seg_dx = seg_dx / segment_len
            norm_seg_dy = seg_dy / segment_len
        
        trace_label_offset = 12  # Distance from the start of the segment along its path
                                 # Adjust this value based on node size and desired placement

        # Only draw trace label if segment is long enough and not a zero-length segment
        if segment_len > trace_label_offset :
            trace_x = node_positions[from_node, 0] + norm_seg_dx * trace_label_offset
            trace_y = node_positions[from_node, 1] + norm_seg_dy * trace_label_offset
            
            bbox_props_trace = dict(boxstyle="circle,pad=0.1", facecolor='black', alpha=0.6, edgecolor='yellow')
            ax1.text(trace_x, trace_y, f"#{i+1}", fontsize=7, ha='center', va='center',
                     color='yellow', weight='bold', bbox=bbox_props_trace, zorder=5) # Ensure it's on top
        # ====================================================================

    # Nodes
    for i in range(Config.NUM_NODES):
        if i == start_node:
            ax1.scatter(node_positions[i, 0], node_positions[i, 1], 
                       s=400, c='lime', marker='*', edgecolors='white', linewidths=2, zorder=3)
            ax1.text(node_positions[i, 0], node_positions[i, 1]-8, f'START\n{i}', 
                    ha='center', va='top', fontsize=10, weight='bold', color='lime', zorder=4)
        elif i in target_nodes:
            is_on_route = i in actual_route
            node_color = 'magenta' if is_on_route else 'cyan' # Highlight target nodes on the route differently if desired
            ax1.scatter(node_positions[i, 0], node_positions[i, 1], 
                       s=250, c=node_color, marker='s', edgecolors='white', linewidths=2, zorder=3)
            ax1.text(node_positions[i, 0], node_positions[i, 1]-6, str(i), 
                    ha='center', va='top', fontsize=9, weight='bold', color=node_color, zorder=4)
        else:
            ax1.scatter(node_positions[i, 0], node_positions[i, 1], 
                       s=100, c='gray', marker='o', edgecolors='white', linewidths=1, zorder=3)
            ax1.text(node_positions[i, 0], node_positions[i, 1]-4, str(i), 
                    ha='center', va='top', fontsize=8, color='gray', zorder=4)

    legend_elements = [
        plt.Line2D([0], [0], color='#44ff44', lw=2, label='Comfort ≥ 16 (High)'),
        plt.Line2D([0], [0], color='#ffaa44', lw=2, label='Comfort ≥ 12 (Mid)'),
        plt.Line2D([0], [0], color='#ff4444', lw=2, label='Comfort < 12 (Low)'),
        plt.Line2D([0], [0], marker='o', color='w', label='Trace Number (#N)', # Legend for trace marker
                   markerfacecolor='yellow', markersize=5, markeredgecolor='black', linestyle='None')
    ]
    ax1.legend(handles=legend_elements, fontsize='small', loc='best')
    
    ax1.set_title(f"Best Route (Distance: {total_distance_val:.1f}, Avg Comfort: {avg_comfort_val:.2f})", 
                  fontsize=16, weight='bold', color='white', pad=20)
    ax1.set_xlabel("X Coordinate", color='white') # Added axis labels
    ax1.set_ylabel("Y Coordinate", color='white') # Added axis labels
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#1a1a1a')
    
    # Pareto Front
    if len(hof[0].fitness.weights) > 1:
        ax2 = fig.add_subplot(gs[0, 1])
        pareto_distances = np.array([ind.fitness.values[0] for ind in hof])
        pareto_comforts = np.array([-ind.fitness.values[1] for ind in hof]) # Convert back from minimized neg comfort
        
        scatter = ax2.scatter(pareto_distances, pareto_comforts, c=pareto_comforts, cmap='viridis', 
                             s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
        
        # Highlight best compromise on Pareto plot
        ax2.scatter(total_distance_val, avg_comfort_val, 
                   s=200, c='red', marker='*', edgecolors='white', linewidths=2, zorder=5, label='Selected Best')
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Average Comfort', color='white')
        cbar.ax.yaxis.set_tick_params(color='white') # Colorbar ticks
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white') # Colorbar tick labels

        ax2.set_xlabel('Total Distance', fontsize=12, color='white')
        ax2.set_ylabel('Average Comfort', fontsize=12, color='white')
        ax2.set_title('Pareto Front Analysis', fontsize=14, weight='bold', color='white')
        ax2.tick_params(axis='x', colors='white') # Tick colors
        ax2.tick_params(axis='y', colors='white') # Tick colors
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize='small')
        ax2.set_facecolor('#1a1a1a')
    
    # Summary Statistics 
    ax3 = fig.add_subplot(gs[1, 1])
    route_str = ' → '.join(map(str, actual_route))
    if len(route_str) > 60: # Truncate long route strings for display
        route_str = route_str[:57] + "..."

    stats_text = f"""
    OPTIMAL ROUTE STATISTICS
    
    Route: {route_str}
    
    Total Distance: {total_distance_val:.1f}
    Average Comfort: {avg_comfort_val:.2f}
    Min Edge Comfort: {min_route_comfort:.2f}
    Max Edge Comfort: {max_route_comfort:.2f}
    
    Segments: {len(actual_route)-1}
    Targets Visited: {Config.TARGET_COUNT}
    
    GA Config:
      Pop: {Config.POPULATION_SIZE}, Gen: {Config.GENERATIONS}
      CX: {Config.CX_PROB}, Mut: {Config.MUT_PROB}
      Comfort: {Config.COMFORT_HANDLING}
    """
    if Config.COMFORT_HANDLING in ['constraint', 'both']:
        stats_text += f"""
      Min Avg Comfort: {Config.MIN_AVG_COMFORT}
      Min Edge Comfort: {Config.MIN_EDGE_COMFORT}
    """
    
    ax3.text(0.05, 0.95, stats_text.strip(), transform=ax3.transAxes, fontsize=9, # Reduced fontsize
            verticalalignment='top', color='white', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#2a2a2a', alpha=0.8))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    #ax3.set_title('Run Summary', fontsize=14, weight='bold', color='white') # Title for this subplot
    
    plt.suptitle('Route Planning Optimization Results', 
                fontsize=20, weight='bold', color='white', y=0.99) # Adjusted y for suptitle
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.savefig("streamlined_route_analysis.png", dpi=300, bbox_inches='tight', 
                facecolor='#1a1a1a', edgecolor='none')
    plt.show()
    return node_positions, target_nodes, start_node # Return values for plot_black_nodes_only


def print_enhanced_route_details(hof, dist_matrix, comfort_matrix, target_nodes, start_node):
    """Print detailed route analysis with enhanced formatting"""
    if not hof:
        print("HOF is empty, cannot print route details.")
        return

    best_individual = None
    if len(hof[0].fitness.weights) > 1: # Multi-objective
        distances = np.array([ind.fitness.values[0] for ind in hof])
        comforts = np.array([-ind.fitness.values[1] for ind in hof]) # fitness has -avg_comfort
        if not distances.size :
            best_individual = hof[0]
        else:
            norm_dist = (distances - distances.min()) / (distances.max() - distances.min()) if distances.max() > distances.min() else np.zeros_like(distances)
            norm_comfort = (comforts.max() - comforts) / (comforts.max() - comforts.min()) if comforts.max() > comforts.min() else np.zeros_like(comforts)
            scores = np.sqrt(norm_dist**2 + norm_comfort**2)
            best_idx = np.argmin(scores)
            best_individual = hof[best_idx]
    else: # Single-objective
        best_individual = hof[0]
    
    actual_route = [start_node] + [target_nodes[i] for i in best_individual] + [start_node]
    
    print("\n" + "="*80)
    print("OPTIMAL ROUTE ANALYSIS")
    print("="*80)
    
    print(f"Route Path: {' → '.join(map(str, actual_route))}")
    print(f"Target Nodes (Configured): {sorted(target_nodes)}")
    print(f"Starting Node: {start_node}")
    
    total_distance = sum(dist_matrix[actual_route[i], actual_route[i+1]] for i in range(len(actual_route)-1))
    
    num_edges = len(actual_route) - 1
    if num_edges > 0:
        avg_comfort = sum(comfort_matrix[actual_route[i], actual_route[i+1]] for i in range(num_edges)) / num_edges
        min_edge_comfort = min(comfort_matrix[actual_route[i], actual_route[i+1]] for i in range(num_edges))
        max_edge_comfort = max(comfort_matrix[actual_route[i], actual_route[i+1]] for i in range(num_edges))
    else:
        avg_comfort = 0
        min_edge_comfort = 0
        max_edge_comfort = 0

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total Distance:     {total_distance:.2f}")
    print(f"Average Comfort:    {avg_comfort:.2f}")
    print(f"Min Edge Comfort:   {min_edge_comfort:.2f}")
    print(f"Max Edge Comfort:   {max_edge_comfort:.2f}")
    print(f"Segments Count:     {num_edges}")
    
    print("\n" + "="*80)
    print("SEGMENT DETAILS:")
    print("="*80)
    print(f"{'Segment':<10} | {'Path':<15} | {'Distance':<10} | {'Comfort':<10}")
    print("-"*50)
    for i in range(num_edges):
        from_n = actual_route[i]
        to_n = actual_route[i+1]
        dist = dist_matrix[from_n, to_n]
        cmf = comfort_matrix[from_n, to_n]
        print(f"#{i+1:<8} | {str(from_n):>5} → {str(to_n):<5} | {dist:<10.1f} | {cmf:<10.1f}")
    print("="*80)


def visualize_results(hof, dist_matrix, comfort_matrix, target_nodes, start_node):
    """Streamlined visualization of the best route, Pareto front and route statistics"""
    
    print("\nCreating Streamlined Visualizations...")
    
    node_positions, returned_target_nodes, returned_start_node = create_streamlined_visualization(
        hof, dist_matrix, comfort_matrix, target_nodes, start_node
    )
    print("Streamlined visualization created successfully!")

    # Use returned values from create_streamlined_visualization as they might be internally generated
    plot_black_nodes_only(node_positions, returned_target_nodes, returned_start_node)
    print_enhanced_route_details(hof, dist_matrix, comfort_matrix, target_nodes, start_node)


# ======================
# Main Function
# ======================
def main():
    dist_matrix, comfort_matrix, target_nodes = generate_data()
    
    print("="*50)
    print("Route Planning Parameter Configuration:")
    print(f"Total Nodes: {Config.NUM_NODES}")
    print(f"Start Node: {Config.START_NODE}")
    print(f"Target Nodes To Visit: {Config.TARGET_COUNT} (Actual: {target_nodes})")
    print(f"Comfort Handling Method: {Config.COMFORT_HANDLING}")
    if Config.COMFORT_HANDLING in ['constraint', 'both', 'desirability']: # Desirability can also use penalty
        print(f"Minimum Edge Comfort: {Config.MIN_EDGE_COMFORT}")
        print(f"Minimum Average Comfort: {Config.MIN_AVG_COMFORT}")
    print("="*50)
    
    toolbox = setup_algorithm(dist_matrix, comfort_matrix, target_nodes, Config.START_NODE)
    
    population = toolbox.population(n=Config.POPULATION_SIZE)
    
    # Use ParetoFront for multi-objective, HallOfFame for single-objective
    if len(creator.Fitness.weights) > 1:
        hof = tools.ParetoFront()
    else:
        hof = tools.HallOfFame(1) # Store the single best for single-objective
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    if len(toolbox.individual().fitness.weights) > 1:
        stats.register("avg_dist", lambda x: np.mean([val[0] for val in x]))
        stats.register("avg_neg_comfort", lambda x: np.mean([val[1] for val in x])) # avg of -avg_comfort
        stats.register("min_dist", lambda x: min(val[0] for val in x))
        stats.register("max_comfort", lambda x: -min(val[1] for val in x)) # max of avg_comfort
    else: # Single objective
        stats.register("avg", np.mean)
        stats.register("min", np.min) # We minimize the fitness value
    
    # Using eaMuPlusLambda as in the original snippet, ensure mu and lambda_ are sensible.
    # mu: number of individuals to select for the next generation.
    # lambda_: number of children to produce at each generation.
    # A common setup is mu < lambda_ or mu = lambda_.
    # If lambda_ = Config.GENERATIONS, this is unusual. Typically lambda_ is POPULATION_SIZE or 2*POPULATION_SIZE.
    # Let's set lambda_ = Config.POPULATION_SIZE for a (mu + lambda) strategy where mu=lambda.
    # Or, if eaSimple was intended:
    # algorithms.eaSimple(population, toolbox, cxpb=Config.CX_PROB, mutpb=Config.MUT_PROB, ngen=Config.GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    algorithms.eaMuPlusLambda(
        population, toolbox, 
        mu=Config.POPULATION_SIZE, 
        lambda_=Config.POPULATION_SIZE * 2, # Produce 2x population size offspring
        cxpb=Config.CX_PROB, mutpb=Config.MUT_PROB,
        ngen=Config.GENERATIONS, stats=stats,
        halloffame=hof, verbose=True
    )
    
    if not hof:
        print("Optimization finished, but Hall of Fame is empty. No solutions found or recorded.")
    else:
        print(f"Optimization finished. Hall of Fame contains {len(hof)} individual(s).")
        if len(creator.Fitness.weights) == 1 and len(hof) > 0:
            print(f"Best single-objective solution fitness: {hof[0].fitness.values}")

    visualize_results(hof, dist_matrix, comfort_matrix, target_nodes, Config.START_NODE)

if __name__ == "__main__":
    main()