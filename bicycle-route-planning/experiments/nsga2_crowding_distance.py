import numpy as np
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from functools import partial
# import pygmo as pg # If you were using it for custom HV selection
import copy

# ======================
# Configuration Parameters
# ======================
class Config:
    # Network parameters
    NUM_NODES =20
    START_NODE = 0
    TARGET_COUNT = 8 # Number of actual target nodes to visit in order

    # Algorithm parameters
    POPULATION_SIZE = 100 # mu
    OFFSPRING_SIZE = 100  # lambda (Added for clarity with eaMuPlusLambda)
    GENERATIONS = 100
    CX_PROB = 0.6
    MUT_PROB = 0.3

    COMFORT_HANDLING = 'both'

    MIN_AVG_COMFORT_THRESHOLD = 3.0 # Renamed from MIN_AVG_COMFORT
    MIN_EDGE_COMFORT_THRESHOLD = 2.0  # Renamed from MIN_EDGE_COMFORT
    COMFORT_PENALTY = 10000
    # HV_REF_POINT = [20000.0, 0.0] # If using custom HV selection

# ======================
# Data Generation
# ======================
def generate_data():
    np.random.seed(42); random.seed(42)
    dist_matrix = np.random.randint(10, 100, size=(Config.NUM_NODES, Config.NUM_NODES))
    # comfort_matrix is now edge_comfort_scores in the rest of the code
    edge_comfort_scores = np.random.randint(1, 6, size=(Config.NUM_NODES, Config.NUM_NODES))
    for i in range(Config.NUM_NODES):
        for j in range(i, Config.NUM_NODES): # Corrected loop to avoid overwrite and ensure symmetry
            if i == j: dist_matrix[i][j] = 0; edge_comfort_scores[i][j] = 0
            else:
                dist_matrix[i][j] = dist_matrix[j][i]
                edge_comfort_scores[i][j] = edge_comfort_scores[j][i] # Symmetry for comfort
    all_nodes = list(range(Config.NUM_NODES)); all_nodes.remove(Config.START_NODE)
    # Use TARGET_COUNT from Config
    num_targets = Config.TARGET_COUNT
    if num_targets > len(all_nodes): # Ensure we don't try to sample more than available
        num_targets = len(all_nodes)
    target_nodes = random.sample(all_nodes, num_targets)
    return dist_matrix, edge_comfort_scores, target_nodes # Return edge_comfort_scores

# ======================
# Evaluation Function
# ======================
def evaluate_route(individual, dist_matrix, edge_comfort_scores, target_nodes, start_node): # Use edge_comfort_scores
    actual_route = [start_node] + [target_nodes[i] for i in individual] + [start_node]
    total_distance, total_comfort, constraint_violation = 0,0,0
    min_comfort_val = float('inf')
    num_edges = len(actual_route) - 1
    if num_edges == 0:
        if Config.COMFORT_HANDLING == 'objective' or Config.COMFORT_HANDLING == 'both': return float('inf'), float('-inf')
        else: return (float('inf'),)
    for i in range(num_edges):
        from_node, to_node = actual_route[i], actual_route[i+1]
        total_distance += dist_matrix[from_node][to_node]
        current_edge_comfort = edge_comfort_scores[from_node][to_node] # Use edge_comfort_scores
        total_comfort += current_edge_comfort
        if current_edge_comfort < min_comfort_val: min_comfort_val = current_edge_comfort
        if Config.COMFORT_HANDLING in ['constraint', 'both'] and current_edge_comfort < Config.MIN_EDGE_COMFORT_THRESHOLD:
            constraint_violation += 1
    avg_comfort = total_comfort / num_edges if num_edges > 0 else 0
    if Config.COMFORT_HANDLING in ['constraint', 'both'] and avg_comfort < Config.MIN_AVG_COMFORT_THRESHOLD:
        constraint_violation += 1
    penalty = Config.COMFORT_PENALTY * constraint_violation
    if Config.COMFORT_HANDLING == 'objective': return total_distance, avg_comfort
    elif Config.COMFORT_HANDLING == 'constraint': return (total_distance + penalty,)
    else: return total_distance + penalty, avg_comfort

# ======================
# Algorithm Setup
# ======================
def setup_algorithm(dist_matrix, edge_comfort_scores, target_nodes, start_node): # Use edge_comfort_scores
    if hasattr(creator, "Fitness"): del creator.Fitness
    if hasattr(creator, "Individual"): del creator.Individual
    if Config.COMFORT_HANDLING == 'objective': weights = (-1.0, 1.0)
    elif Config.COMFORT_HANDLING == 'constraint': weights = (-1.0,)
    else: weights = (-1.0, 1.0)
    creator.create("Fitness", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.Fitness)
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(target_nodes)), len(target_nodes))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", partial(evaluate_route,
                                        dist_matrix=dist_matrix,
                                        edge_comfort_scores=edge_comfort_scores, # Pass edge_comfort_scores
                                        target_nodes=target_nodes,
                                        start_node=start_node))
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    if len(weights) > 1: toolbox.register("select", tools.selNSGA2)
    else: toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

# ======================
# Result Visualization (MODIFIED plot_route_map and visualize_results call)
# ======================
def plot_pareto_front(pareto_front_individuals, title_suffix=""):
    if not pareto_front_individuals: print("Pareto front empty for plotting."); return
    valid_individuals = [ind for ind in pareto_front_individuals if ind.fitness.valid and len(ind.fitness.values)==2]
    if not valid_individuals: print("No valid MO inds in Pareto front for plot."); return
    obj1_values = [ind.fitness.values[0] for ind in valid_individuals]
    obj2_values = [ind.fitness.values[1] for ind in valid_individuals]
    plt.scatter(obj1_values, obj2_values, c='royalblue', marker='o', label="Pareto Front Solutions", alpha=0.7, s=60, edgecolors='k', linewidth=0.5)
    plt.xlabel("Objective 1 (e.g., Total Distance + Penalty)")
    plt.ylabel("Objective 2 (e.g., Average Comfort)")
    plt.title(f"Pareto Front{title_suffix}")
    plt.grid(linestyle='--', alpha=0.7)

def plot_route_map(individual_to_plot, dist_matrix, edge_comfort_scores, # Use edge_comfort_scores
                        specific_target_nodes, start_node, node_positions, title_suffix=""):
    if not individual_to_plot or not individual_to_plot.fitness.valid:
        print(f"Individual for map is invalid or None: {individual_to_plot}")
        plt.text(0.5, 0.5, "No valid route to display.", ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f"Route Map{title_suffix}")
        return

    route = [start_node]
    for index in individual_to_plot: route.append(specific_target_nodes[index])
    route.append(start_node)

    plt.scatter(node_positions[:, 0], node_positions[:, 1], c='lightgrey', s=40, label="Other Nodes", alpha=0.5)
    plt.scatter(node_positions[start_node, 0], node_positions[start_node, 1],
                c='forestgreen', s=180, marker='*', label=f"Start ({start_node})", zorder=5, edgecolors='black')
    if specific_target_nodes:
        all_target_coords = node_positions[specific_target_nodes]
        plt.scatter(all_target_coords[:, 0], all_target_coords[:, 1],
                    c='salmon', s=100, marker='s', label="Defined Targets", zorder=2, alpha=0.6)
        for node_idx in specific_target_nodes:
             plt.text(node_positions[node_idx, 0] + 1.5, node_positions[node_idx, 1] + 1.5, f"T{node_idx}", fontsize=8, color='darkred', alpha=0.8)
    target_nodes_in_this_route_actual = [specific_target_nodes[i] for i in individual_to_plot]
    if target_nodes_in_this_route_actual:
        target_node_coords_in_route = node_positions[target_nodes_in_this_route_actual]
        plt.scatter(target_node_coords_in_route[:, 0], target_node_coords_in_route[:, 1],
                    c='red', s=100, marker='s', zorder=4, edgecolors='black') # Highlight targets in route

    # --- Custom Legend for Comfort Colors ---
    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=2, label='Comfort >= 4 (High)'),
        plt.Line2D([0], [0], color='orange', lw=2, label='Comfort >= 2.5 (Mid)'),
        plt.Line2D([0], [0], color='red', lw=2, label='Comfort < 2.5 (Low)')
    ]
    # Store existing legend handles and labels to add them back
    handles, labels = plt.gca().get_legend_handles_labels()
    # --- End Custom Legend Prep ---

    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        start_pos, end_pos = node_positions[u], node_positions[v]
        comfort = edge_comfort_scores[u,v] # Use edge_comfort_scores
        color = 'green' if comfort >= 4 else ('orange' if comfort >= 2.5 else 'red')
        plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                 color=color, linewidth=2.5, alpha=0.75, zorder=3)
        dist_val = dist_matrix[u,v]
        plt.text((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2 + 1,
                 f"{dist_val}", fontsize=7, ha='center', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.7, pad=0.5, boxstyle="round,pad=0.1"))

    for i, node_idx_in_targets_list in enumerate(individual_to_plot):
        actual_node_idx = specific_target_nodes[node_idx_in_targets_list]
        plt.text(node_positions[actual_node_idx, 0], node_positions[actual_node_idx, 1] - 5,
                 f"#{i+1}", fontsize=10, color='black', ha='center', weight='bold',
                 bbox=dict(facecolor='lightyellow', alpha=0.8, pad=0.1, boxstyle="circle,pad=0.2"))

    plt.xlabel("X-coordinate"); plt.ylabel("Y-coordinate"); plt.title(f"Route Map{title_suffix}")
    # --- Add comfort legend elements to existing ones ---
    all_handles = handles + legend_elements
    all_labels = labels + [le.get_label() for le in legend_elements]
    plt.legend(all_handles, all_labels, fontsize='small', loc='best')
    # --- End Legend Update ---
    plt.axis('equal'); plt.grid(False)


def visualize_results(solutions_archive, dist_matrix, edge_comfort_scores, # Renamed
                      specific_target_nodes, start_node): # Renamed
    if not solutions_archive:
        print("No solutions to visualize."); plt.figure(figsize=(8, 5)); plt.text(0.5, 0.5, "No solutions found.", ha='center', va='center'); plt.title("Optimization Results"); plt.show()
        return

    np.random.seed(42); node_positions = np.random.rand(Config.NUM_NODES, 2) * 100
    plt.figure(figsize=(17, 7)); solution_for_map = None; map_title_suffix = " (No specific solution selected for map)"
    is_multi_objective = len(solutions_archive[0].fitness.weights) > 1

    if is_multi_objective:
        plt.subplot(1, 2, 1); plot_pareto_front(solutions_archive, " (NSGA-II)")
        valid_pareto_solutions = [s for s in solutions_archive if s.fitness.valid and len(s.fitness.values) == 2]
        if valid_pareto_solutions:
            obj1_vals = np.array([ind.fitness.values[0] for ind in valid_pareto_solutions]); obj2_vals = np.array([ind.fitness.values[1] for ind in valid_pareto_solutions])
            if len(obj1_vals) > 0:
                min_o1, max_o1 = obj1_vals.min(), obj1_vals.max(); min_o2, max_o2 = obj2_vals.min(), obj2_vals.max(); epsilon = 1e-9
                range_o1 = (max_o1 - min_o1) if (max_o1 - min_o1) > epsilon else epsilon; range_o2 = (max_o2 - min_o2) if (max_o2 - min_o2) > epsilon else epsilon
                norm_obj1 = (obj1_vals - min_o1) / range_o1; norm_obj2 = (obj2_vals - min_o2) / range_o2
                distances_to_utopian = np.sqrt(norm_obj1**2 + (1 - norm_obj2)**2); compromise_idx = np.argmin(distances_to_utopian)
                solution_for_map = valid_pareto_solutions[compromise_idx]
                plt.scatter(solution_for_map.fitness.values[0], solution_for_map.fitness.values[1], c='red', s=150, marker='*', label="Compromise Solution", zorder=10, edgecolors='black')
        plt.legend() # Ensure legend for Pareto plot is shown
    else:
        plt.subplot(1, 2, 1); plt.text(0.5, 0.5, "Single Objective Opt", ha='center', va='center', transform=plt.gca().transAxes); plt.title("Optimization Mode")
        if solutions_archive: solution_for_map = solutions_archive[0]

    plt.subplot(1, 2, 2) # This selects/creates the second subplot for the map
    if solution_for_map:
        _original_comfort_handling = Config.COMFORT_HANDLING; Config.COMFORT_HANDLING = 'objective'
        eval_values_clean = evaluate_route(solution_for_map, dist_matrix, edge_comfort_scores, specific_target_nodes, start_node)
        Config.COMFORT_HANDLING = _original_comfort_handling
        obj1_stored = solution_for_map.fitness.values[0]; clean_dist_display = f"{eval_values_clean[0]:.0f}"
        clean_comfort_display = f"{eval_values_clean[1]:.2f}" if isinstance(eval_values_clean, tuple) and len(eval_values_clean) > 1 else "N/A"
        if is_multi_objective:
            obj2_stored_display = f"{solution_for_map.fitness.values[1]:.2f}"
            map_title_suffix = (f" (Compromised: Dist={clean_dist_display}, Comfort={clean_comfort_display})")
        else:
            map_title_suffix = (f" (Best: Stored Obj1={obj1_stored:.0f}\n"
                                f"Clean Eval: Dist={clean_dist_display}, Comfort={clean_comfort_display})")
        plot_route_map(solution_for_map, dist_matrix, edge_comfort_scores,
                            specific_target_nodes, start_node, node_positions, map_title_suffix)
    else:
        plt.text(0.5, 0.5, "No solution for route map.", ha='center', va='center', transform=plt.gca().transAxes); plt.title("Route Map")
    plt.tight_layout(pad=2.5)
    file_suffix = Config.COMFORT_HANDLING;
    if is_multi_objective: file_suffix += "_NSGA2"
    plt.savefig(f"route_planning_results_{file_suffix}.png", dpi=150); plt.show()
    if solution_for_map: print_route_details(solution_for_map, dist_matrix, edge_comfort_scores, specific_target_nodes, start_node)

def print_route_details(individual, dist_matrix, edge_comfort_scores, specific_target_nodes, start_node): # Use edge_comfort_scores
    route=[start_node]; [route.append(specific_target_nodes[i]) for i in individual]; route.append(start_node)
    print("\n" + "="*10 + " Sel Route Details " + "="*10); print(f"Ind (Permutation Target Indices): {list(individual)}"); print(f"Target Order: {[specific_target_nodes[i] for i in individual]}"); print(f"Full Route: {' -> '.join(map(str,route))}")
    tdc,sec,ne,cvp = 0,0,len(route)-1,0; print("\nSegment Details:")
    avg_comfort_for_print_calc = sum(edge_comfort_scores[route[i],route[i+1]] for i in range(ne)) / ne if ne > 0 else 0
    for i in range(ne):
        u,v=route[i],route[i+1];d=dist_matrix[u,v];c=edge_comfort_scores[u,v];tdc+=d;sec+=c;s="" # Use edge_comfort_scores
        if Config.COMFORT_HANDLING in ['constraint','both']:
            if c < Config.MIN_EDGE_COMFORT_THRESHOLD: s+=" [EDGE COMFORT VIOLATION!]";cvp+=1
    # Check avg comfort violation after calculating total, not in loop per segment
    for i in range(ne): # Second loop just for printing with correct status
        u,v=route[i],route[i+1];d=dist_matrix[u,v];c=edge_comfort_scores[u,v];s=""
        if Config.COMFORT_HANDLING in ['constraint','both']:
            if c < Config.MIN_EDGE_COMFORT_THRESHOLD: s+=" [EDGE COMFORT VIOLATION!]"
            if i == ne -1 and avg_comfort_for_print_calc < Config.MIN_AVG_COMFORT_THRESHOLD: # Check only once for the whole route
                s += " (Overall AVG COMFORT VIOLATION!)"
        print(f"  {u}->{v}: Dist={d}, EdgeComfort={c}{s}")

    print(f"\nAverage Path Comfort (calculated for print): {avg_comfort_for_print_calc:.2f}")
    if Config.COMFORT_HANDLING in ['constraint','both'] and avg_comfort_for_print_calc < Config.MIN_AVG_COMFORT_THRESHOLD and not "[AVG COMFORT VIOLATION!]" in s: # if not already printed
        print("⚠️ Warning: Average route comfort does not meet the requirement!")
    print(f"Total Calculated Distance (from segments): {tdc}")
    print(f"\nFitness values from DEAP individual: {individual.fitness.values}")
    if Config.COMFORT_HANDLING in ['constraint','both']:
        recalculated_penalty = Config.COMFORT_PENALTY * cvp # cvp counts edge violations only for now
        # Recalculate constraint violations fully for print
        final_cvp_print = 0
        if any(edge_comfort_scores[route_i][route_i+1] < Config.MIN_EDGE_COMFORT_THRESHOLD for route_i in range(ne)): final_cvp_print +=1 # Simplified: counts as 1 if any edge fails
        if avg_comfort_for_print_calc < Config.MIN_AVG_COMFORT_THRESHOLD: final_cvp_print +=1

        print(f"Recalculated Constraint Violations for this route (edge/avg): {final_cvp_print}") # This cvp is based on printed logic
        print(f"Recalculated Penalty for this route: {Config.COMFORT_PENALTY * final_cvp_print}")
        print(f"Recalculated (Dist + Penalty): {tdc + Config.COMFORT_PENALTY * final_cvp_print}")
    print("="*50)

# ======================
# Main Function
# ======================
def main():
    dist_matrix, comfort_matrix, target_nodes = generate_data() # comfort_matrix is edge_comfort_scores
    print("="*50); print("Route Planning Parameter Configuration:"); print("="*50)
    print(f"Total Nodes: {Config.NUM_NODES}, Start: {Config.START_NODE}, Targets: {target_nodes}")
    print(f"Comfort Handling: {Config.COMFORT_HANDLING}")
    if Config.COMFORT_HANDLING in ['constraint', 'both']: print(f"  Min Edge Comfort: {Config.MIN_EDGE_COMFORT_THRESHOLD}, Min Avg Comfort: {Config.MIN_AVG_COMFORT_THRESHOLD}")
    print(f"Pop: {Config.POPULATION_SIZE}, Offspring: {getattr(Config, 'OFFSPRING_SIZE', Config.POPULATION_SIZE)}, Gens: {Config.GENERATIONS}") # Check if OFFSPRING_SIZE exists
    print(f"CX_PROB: {Config.CX_PROB}, MUT_PROB: {Config.MUT_PROB}"); print("="*50)
    toolbox = setup_algorithm(dist_matrix, comfort_matrix, target_nodes, Config.START_NODE)
    population = toolbox.population(n=Config.POPULATION_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    is_multi_objective_run = len(toolbox.individual().fitness.weights) > 1
    if is_multi_objective_run:
        stats.register("avg_o1", lambda x: np.mean([fit[0] for fit in x if fit and len(fit)>0 and fit[0]!=float('inf')]) if x and any(fit and len(fit)>0 and fit[0]!=float('inf') for fit in x) else np.nan)
        stats.register("avg_o2", lambda x: np.mean([fit[1] for fit in x if fit and len(fit)>1 and fit[1]!=float('-inf')]) if x and any(fit and len(fit)>1 and fit[1]!=float('-inf') for fit in x) else np.nan)
    else: stats.register("avg", np.mean); stats.register("min", np.min)
    archive = tools.ParetoFront() if is_multi_objective_run else tools.HallOfFame(1)
    if is_multi_objective_run: print("Using tools.ParetoFront for archiving.")
    else: print("Using tools.HallOfFame(1) for archiving.")
    current_offspring_size = getattr(Config, 'OFFSPRING_SIZE', Config.POPULATION_SIZE)
    if is_multi_objective_run:
        final_population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=Config.POPULATION_SIZE, lambda_=current_offspring_size, cxpb=Config.CX_PROB, mutpb=Config.MUT_PROB, ngen=Config.GENERATIONS, stats=stats, halloffame=archive, verbose=True)
    else:
        final_population, logbook = algorithms.eaSimple(population, toolbox, cxpb=Config.CX_PROB, mutpb=Config.MUT_PROB, ngen=Config.GENERATIONS, stats=stats, halloffame=archive, verbose=True)
    print("\n" + "="*50); print("Optimization Finished."); print("="*50)
    solutions_for_viz = list(archive)
    if is_multi_objective_run: print(f"Final Pareto Front Size: {len(solutions_for_viz)}")
    else:
        if solutions_for_viz: print(f"Best Solution Fitness: {solutions_for_viz[0].fitness.values}")
    visualize_results(solutions_for_viz, dist_matrix, comfort_matrix, target_nodes, Config.START_NODE)

if __name__ == "__main__":
    main()