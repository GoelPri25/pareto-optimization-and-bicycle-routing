import numpy as np
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from functools import partial
import pygmo as pg # For hypervolume calculation
import copy # For deep copying individuals

# ======================
# Configuration Parameters
# ======================
class Config:
    NUM_NODES = 20
    START_NODE = 0
    TARGET_NODES_COUNT = 8 # Renamed for clarity in generate_data
    POPULATION_SIZE = 100  # mu (Smaller for HV-based selection due to cost)
    OFFSPRING_SIZE = 100   # lambda
    GENERATIONS = 50 # Might need fewer generations if convergence is faster or cost is high
    CX_PROB = 0.8
    MUT_PROB = 0.2
    COMFORT_HANDLING = 'both'
    MIN_AVG_COMFORT_THRESHOLD = 3.0
    MIN_EDGE_COMFORT_THRESHOLD = 2.0
    COMFORT_PENALTY = 10000
    # HV_REF_POINT: Must be set carefully!
    # Obj1 = Dist+Penalty (minimize), Obj2 = AvgComfort (maximize)
    # For pygmo: Obj1_hv = Dist+Penalty, Obj2_hv = -AvgComfort
    # HV_REF_POINT = [Ref_for_Obj1_hv, Ref_for_Obj2_hv]
    # Ref_for_Obj1_hv > max(Obj1_hv), Ref_for_Obj2_hv > max(Obj2_hv) (i.e. > max(-AvgComfort))
    # If Dist+Pen ~200-1000, AvgComfort ~1-5 (-AvgComfort ~ -5 to -1)
    HV_REF_POINT = [30000, 0.0] # Example: [max_dist_penalty_bound, better_than_best_-comfort_bound]

# ======================
# Data Generation
# ======================
def generate_data():
    np.random.seed(42); random.seed(42)
    dist_matrix = np.random.randint(10, 100, size=(Config.NUM_NODES, Config.NUM_NODES))
    edge_comfort_scores = np.random.randint(1, 6, size=(Config.NUM_NODES, Config.NUM_NODES))
    for i in range(Config.NUM_NODES):
        for j in range(i, Config.NUM_NODES):
            if i == j: dist_matrix[i][j] = 0; edge_comfort_scores[i][j] = 0
            else: dist_matrix[i][j] = dist_matrix[j][i]; edge_comfort_scores[i][j] = edge_comfort_scores[j][i]
    all_nodes_except_start = list(range(Config.NUM_NODES)); all_nodes_except_start.remove(Config.START_NODE)
    num_specific_targets = Config.TARGET_NODES_COUNT
    if num_specific_targets > len(all_nodes_except_start): num_specific_targets = len(all_nodes_except_start)
    specific_target_nodes = random.sample(all_nodes_except_start, num_specific_targets)
    return dist_matrix, edge_comfort_scores, specific_target_nodes

# ======================
# Evaluation Function (same)
# ======================
def evaluate_route(individual_permutation_indices, dist_matrix, edge_comfort_scores,
                   specific_target_nodes, start_node):
    route = [start_node]; [route.append(specific_target_nodes[i]) for i in individual_permutation_indices]; route.append(start_node)
    total_distance, sum_edge_comforts, num_edges, constraint_violations = 0,0,len(route)-1,0
    if num_edges == 0:
        if Config.COMFORT_HANDLING == 'objective' or Config.COMFORT_HANDLING == 'both': return float('inf'), float('-inf')
        else: return (float('inf'),)
    for i in range(num_edges):
        u, v = route[i], route[i+1]; total_distance += dist_matrix[u][v]
        current_edge_comfort = edge_comfort_scores[u][v]; sum_edge_comforts += current_edge_comfort
        if Config.COMFORT_HANDLING in ['constraint', 'both'] and current_edge_comfort < Config.MIN_EDGE_COMFORT_THRESHOLD:
            constraint_violations += 1
    avg_path_comfort = sum_edge_comforts / num_edges if num_edges > 0 else 0.0
    if Config.COMFORT_HANDLING in ['constraint', 'both'] and avg_path_comfort < Config.MIN_AVG_COMFORT_THRESHOLD:
        constraint_violations += 1
    penalty = Config.COMFORT_PENALTY * constraint_violations
    if Config.COMFORT_HANDLING == 'objective': return total_distance, avg_path_comfort
    elif Config.COMFORT_HANDLING == 'constraint': return (total_distance + penalty,)
    else: return total_distance + penalty, avg_path_comfort

# ======================
# Hypervolume Helper (using pygmo)
# ======================
def get_hypervolume_value(objective_vectors_for_hv, ref_point):
    """Calculates hypervolume for a list of objective vectors [ [o1,o2], [o1,o2] ...]."""
    if not objective_vectors_for_hv: return 0.0
    try:
        hv_calculator = pg.hypervolume(objective_vectors_for_hv)
        return hv_calculator.compute(ref_point)
    except Exception as e:
        # print(f"Error in get_hypervolume_value: {e} with points {objective_vectors_for_hv} and ref {ref_point}")
        return 0.0 # Error or invalid points (e.g., dominated by ref_point)

def get_hv_points_from_individuals(individuals):
    """Converts DEAP individuals' fitness values to points for HV calculation."""
    points = []
    if not individuals: return points
    is_multi_obj = len(individuals[0].fitness.weights) > 1
    for ind in individuals:
        if ind.fitness.valid:
            if is_multi_obj and len(ind.fitness.values) == 2:
                points.append([ind.fitness.values[0], -ind.fitness.values[1]]) # [Obj1_min, -Obj2_max_as_min]
            elif not is_multi_obj and len(ind.fitness.values) == 1:
                 # For 1D HV, often a dummy second objective is used, or handle differently
                 points.append([ind.fitness.values[0], 0]) # Assuming ref point has 2 dims
    return points

# ======================
# Custom Hypervolume-Based Selection Operator
# ======================
def selGreedyHypervolume(individuals, k, ref_point):
    """
    Selects k individuals by greedily maximizing hypervolume.
    1. Takes all non-dominated individuals from the input set.
    2. If more than k non-dominated, greedily selects k that maximize HV.
    3. If less than k, takes all non-dominated and fills with dominated ones
       (this part can be improved, e.g. by using NSGA-II's front filling).
       For simplicity, this version will just take the best HV non-dominated if |F1| > k,
       or all of F1 if |F1| <= k.
    """
    if not individuals: return []
    if k == 0: return []

    # Ensure all individuals have valid fitness
    valid_individuals = [ind for ind in individuals if ind.fitness.valid]
    if not valid_individuals: return []

    # 1. Find the first non-dominated front
    #    sortNondominated returns a list of fronts; we take the first one.
    #    It needs the number of individuals to keep, but we want all non-dominated from the current pool.
    #    So, we pass len(valid_individuals) to get all fronts, then take fronts[0].
    #    Or, use tools.sortNondominated with first_front_only=True (DEAP versions vary)
    #    A simpler way for just the first front:
    current_front = tools.selBest(valid_individuals, len(valid_individuals), fit_attr="fitness") # Gets all if k=len
    current_front = tools.emo.sortNondominated(current_front, len(current_front))[0]


    if not current_front: return []

    if len(current_front) <= k:
        # If the first front has k or fewer individuals, select all of them.
        # Optionally, fill remaining spots from subsequent fronts (more complex)
        # For now, just return the first front if it's small enough or all we have.
        # If you need to strictly return k, you'd need to add dominated solutions,
        # which is usually not what HV selection does primarily.
        # print(f"  selGreedyHypervolume: Front size {len(current_front)} <= k={k}. Returning front.")
        return current_front
    else:
        # If the front is larger than k, we need to select a subset of k points
        # that maximizes hypervolume (greedy approach).
        # print(f"  selGreedyHypervolume: Front size {len(current_front)} > k={k}. Selecting greedily.")
        chosen_for_hv_calc = []
        # Create a copy of the front to manipulate for greedy selection
        candidates = [copy.deepcopy(ind) for ind in current_front] # Deepcopy to avoid modifying original fitness
        
        # Iteratively select k individuals
        for _ in range(k):
            if not candidates: break # No more candidates to choose from

            best_candidate = None
            best_contrib = -1.0 # Smallest possible contribution (HV is non-negative)

            # Find the candidate that adds the most HV to the currently chosen set
            for i, cand_try in enumerate(candidates):
                # Fitness values are already assigned from evaluation
                # We need to ensure they are in the format for get_hypervolume_value
                temp_chosen_fitness_hv_format = get_hv_points_from_individuals(chosen_for_hv_calc + [cand_try])
                current_hv_contrib = get_hypervolume_value(temp_chosen_fitness_hv_format, ref_point)

                # More accurately, we should check the marginal contribution
                # HV(chosen + candidate) - HV(chosen)
                hv_of_current_chosen = get_hypervolume_value(get_hv_points_from_individuals(chosen_for_hv_calc), ref_point)
                marginal_contrib = current_hv_contrib - hv_of_current_chosen

                if marginal_contrib > best_contrib:
                    best_contrib = marginal_contrib
                    best_candidate_idx = i

            if best_candidate_idx is not None:
                chosen_for_hv_calc.append(candidates.pop(best_candidate_idx))
            else: # Should not happen if candidates is not empty
                break
        # print(f"  Greedily selected {len(chosen_for_hv_calc)} individuals for HV.")
        return chosen_for_hv_calc

# ======================
# Algorithm Setup
# ======================
def setup_algorithm(dist_matrix, edge_comfort_scores, specific_target_nodes, start_node):
    fitness_attr_name = "Fitness"; individual_class_name = "Individual"
    if hasattr(creator, fitness_attr_name): delattr(creator, fitness_attr_name)
    if hasattr(creator, individual_class_name): delattr(creator, individual_class_name)
    weights_tuple = None; is_multi_obj = False
    if Config.COMFORT_HANDLING == 'objective' or Config.COMFORT_HANDLING == 'both':
        weights_tuple = (-1.0, 1.0); is_multi_obj = True
    elif Config.COMFORT_HANDLING == 'constraint': weights_tuple = (-1.0,)
    creator.create(fitness_attr_name, base.Fitness, weights=weights_tuple)
    creator.create(individual_class_name, list, fitness=getattr(creator, fitness_attr_name))

    toolbox = base.Toolbox(); num_targets = len(specific_target_nodes)
    toolbox.register("permutation_indices", random.sample, range(num_targets), num_targets)
    toolbox.register("individual", tools.initIterate, getattr(creator, individual_class_name), toolbox.permutation_indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", partial(evaluate_route,
                                        dist_matrix=dist_matrix, edge_comfort_scores=edge_comfort_scores,
                                        specific_target_nodes=specific_target_nodes, start_node=start_node))
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.15) # Slightly increased mutation

    if is_multi_obj:
        toolbox.register("select", selGreedyHypervolume, ref_point=Config.HV_REF_POINT)
        print("Registered selGreedyHypervolume for selection.")
    else:
        toolbox.register("select", tools.selTournament, tournsize=3)
        print("Registered tools.selTournament for selection.")
    return toolbox

# ======================
# Visualization and Print Details
# ======================
def plot_pareto_front(pareto_front_individuals, title_suffix=""):

    if not pareto_front_individuals: print("Pareto front empty for plot."); return
    valid_individuals = [ind for ind in pareto_front_individuals if ind.fitness.valid and len(ind.fitness.values)==2]
    if not valid_individuals: print("No valid MO inds in Pareto front for plot."); return
    obj1 = [ind.fitness.values[0] for ind in valid_individuals]
    obj2 = [ind.fitness.values[1] for ind in valid_individuals]
    plt.scatter(obj1, obj2, c='darkviolet', marker='o', label="Pareto Front (HV-Selected)", alpha=0.7, s=50, edgecolors='k', linewidth=0.5)
    plt.xlabel("Obj1 (Dist+Penalty)"); plt.ylabel("Obj2 (Avg Comfort)")
    plt.title(f"Approximated Pareto Front{title_suffix}"); plt.grid(True, linestyle='--', alpha=0.6)

def plot_route_map(ind, dist_m, edge_cs, spec_targets, start_n, node_pos, title_s=""):

    if not ind or not ind.fitness.valid: print("Ind for map invalid"); plt.text(0.5,0.5,"No valid ind for map.", ha='center'); return
    route = [start_n]; [route.append(spec_targets[i]) for i in ind]; route.append(start_n)
    plt.scatter(node_pos[:,0], node_pos[:,1], c='lightgray',s=40,label="Other Nodes",alpha=0.5)
    plt.scatter(node_pos[start_n,0],node_pos[start_n,1],c='forestgreen',s=150,marker='*',label=f"Start ({start_n})",zorder=5, edgecolors='k')
    actual_targets_in_route = [spec_targets[i] for i in ind]
    if actual_targets_in_route:
        target_coords = node_pos[actual_targets_in_route]
        plt.scatter(target_coords[:,0],target_coords[:,1],c='crimson',s=80,marker='s',label="Targets in Route",zorder=4, edgecolors='k')
        for node_idx in spec_targets: # Label all defined targets for context
             plt.text(node_pos[node_idx, 0] + 1.5, node_pos[node_idx, 1] + 1.5, f"T{node_idx}", fontsize=7, color='darkred', alpha=0.7)
    for i in range(len(route)-1):
        u,v=route[i],route[i+1]; sp,ep=node_pos[u],node_pos[v]; comf=edge_cs[u,v]
        col='green' if comf>=4 else ('orange' if comf >=2.5 else 'red')
        plt.plot([sp[0],ep[0]],[sp[1],ep[1]],color=col,linewidth=2,alpha=0.7,zorder=3)
        plt.text((sp[0]+ep[0])/2,(sp[1]+ep[1])/2+1,f"{dist_m[u,v]}",fontsize=7,ha='center',va='bottom',
                 bbox=dict(facecolor='white',alpha=0.6,pad=0.5,boxstyle="round,pad=0.1"))
    for i,node_idx_in_perm in enumerate(ind): # Iterate through permutation indices
        actual_node_idx = spec_targets[node_idx_in_perm]
        plt.text(node_pos[actual_node_idx,0],node_pos[actual_node_idx,1]-4,f"#{i+1}",fontsize=9,color='k',ha='center',weight='bold',
                 bbox=dict(facecolor='lightyellow',alpha=0.8,pad=0.1,boxstyle="circle,pad=0.2"))
    plt.xlabel("X"); plt.ylabel("Y"); plt.title(f"Route Map{title_s}"); plt.legend(fontsize='small',loc='best'); plt.axis('equal'); plt.grid(False)

def visualize_results(archive_sols, dist_m, edge_cs, spec_targets, start_n):

    if not archive_sols: print("No solutions to visualize."); return
    np.random.seed(42); node_pos = np.random.rand(Config.NUM_NODES,2)*100
    plt.figure(figsize=(17,7)); sol_map=None; map_title_s=" (No sol selected for map)"
    is_mo = len(archive_sols[0].fitness.weights)>1
    if is_mo:
        plt.subplot(1,2,1); plot_pareto_front(archive_sols, " (Greedy HV Selection)")
        valid_p = [s for s in archive_sols if s.fitness.valid and len(s.fitness.values)==2]
        if valid_p:
            o1=[i.fitness.values[0] for i in valid_p]; o2=[i.fitness.values[1] for i in valid_p]
            if o1:
                m1,M1=min(o1),max(o1); m2,M2=min(o2),max(o2); eps=1e-9
                no1=(np.array(o1)-m1)/(M1-m1+eps if M1==m1 else M1-m1)
                no2=(np.array(o2)-m2)/(M2-m2+eps if M2==m2 else M2-m2)
                d_uto = np.sqrt(no1**2+(1-no2)**2); comp_idx=np.argmin(d_uto); sol_map=valid_p[comp_idx]
                plt.scatter(sol_map.fitness.values[0],sol_map.fitness.values[1],c='red',s=150,marker='*',label="Compromise",zorder=10, edgecolors='k')
        plt.legend()
    else:
        plt.subplot(1,2,1); plt.text(0.5,0.5,"Single Obj Opt",ha='center',va='center',transform=plt.gca().transAxes); plt.title("Opt Mode")
        if archive_sols: sol_map=archive_sols[0]
    plt.subplot(1,2,2)
    if sol_map:
        orig_ch=Config.COMFORT_HANDLING; Config.COMFORT_HANDLING='objective'; ev_v=evaluate_route(sol_map,dist_m,edge_cs,spec_targets,start_n); Config.COMFORT_HANDLING=orig_ch
        o1d=sol_map.fitness.values[0]; o2d_stored="N/A"; clean_c_disp = "N/A"
        if is_mo: o2d_stored=f"{sol_map.fitness.values[1]:.2f}"
        if isinstance(ev_v,tuple) and len(ev_v)>1: clean_c_disp=f"{ev_v[1]:.2f}"
        map_title_s=f" (Obj1(stored)={o1d:.0f}, Obj2(stored)={o2d_stored}\nCleanComfort={clean_c_disp})"
        plot_route_map(sol_map,dist_m,edge_cs,spec_targets,start_n,node_pos,map_title_s)
    else: plt.text(0.5,0.5,"No sol for map.",ha='center',va='center',transform=plt.gca().transAxes); plt.title("Route Map")
    plt.tight_layout(pad=2.5); plt.savefig(f"route_results_GreedyHV_{Config.COMFORT_HANDLING}.png",dpi=150); plt.show()
    if sol_map: print_route_details(sol_map,dist_m,edge_cs,spec_targets,start_n)

def print_route_details(ind, dist_m, edge_cs, spec_targets, start_n):

    route=[start_n]; [route.append(spec_targets[i]) for i in ind]; route.append(start_n)
    print("\n"+"="*10+" Sel Route Details "+"="*10); print(f"Ind (Permutation Target Indices): {list(ind)}"); print(f"Target Order: {[spec_targets[i] for i in ind]}"); print(f"Full Route: {' -> '.join(map(str,route))}")
    tdc,sec,ne,cvp=0,0,len(route)-1,0; print("\nSegment Details:")
    for i in range(ne):
        u,v=route[i],route[i+1];d=dist_m[u,v];c=edge_cs[u,v];tdc+=d;sec+=c;s=""
        if Config.COMFORT_HANDLING in ['constraint','both'] and c<Config.MIN_EDGE_COMFORT_THRESHOLD: s+=" [EDGE COMFORT LOW!]";cvp+=1
        print(f"  {u}->{v}: Dist={d}, EdgeComfort={c}{s}")
    apc=sec/ne if ne>0 else 0
    if (Config.COMFORT_HANDLING in ['constraint','both']) and apc<Config.MIN_AVG_COMFORT_THRESHOLD: print(f"Avg Path Comfort: {apc:.2f} [AVG COMFORT LOW!]");cvp+=1
    else: print(f"Avg Path Comfort: {apc:.2f}")
    print(f"Total Calc Dist: {tdc}\nFitness from DEAP: {ind.fitness.values}"); print("="*50)


# ======================
# Main Function
# ======================
def main():
    dist_matrix, edge_comfort_scores, specific_target_nodes = generate_data()
    print("="*50 + "\nRoute Planning Configuration:" + "="*50)
    print(f"Total Nodes: {Config.NUM_NODES}, Start: {Config.START_NODE}, Targets: {specific_target_nodes} (Count: {len(specific_target_nodes)})")
    print(f"Comfort Handling: {Config.COMFORT_HANDLING}")
    if Config.COMFORT_HANDLING in ['constraint', 'both']: print(f"  Min Avg Comfort: {Config.MIN_AVG_COMFORT_THRESHOLD}, Min Edge Comfort: {Config.MIN_EDGE_COMFORT_THRESHOLD}")
    print(f"Pop Size (mu): {Config.POPULATION_SIZE}, Offspring (lambda): {Config.OFFSPRING_SIZE}, Gens: {Config.GENERATIONS}")
    print(f"CX_PROB: {Config.CX_PROB}, MUT_PROB: {Config.MUT_PROB}")
    if Config.COMFORT_HANDLING != 'constraint': print(f"HV Ref Point: {Config.HV_REF_POINT}")
    print("="*50)

    toolbox = setup_algorithm(dist_matrix, edge_comfort_scores, specific_target_nodes, Config.START_NODE)
    population = toolbox.population(n=Config.POPULATION_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    is_multi_obj = len(toolbox.individual().fitness.weights) > 1
    if is_multi_obj:
        stats.register("avg_o1", lambda x: np.mean([fit[0] for fit in x if fit and len(fit)>0 and fit[0]!=float('inf')]) if x else np.nan)
        stats.register("avg_o2", lambda x: np.mean([fit[1] for fit in x if fit and len(fit)>1 and fit[1]!=float('-inf')]) if x else np.nan)
        # Log hypervolume of the archive
        stats.register("archive_hv", lambda pop: get_hypervolume_value(get_hv_points_from_individuals(archive), Config.HV_REF_POINT))
    else:
        stats.register("avg", np.mean); stats.register("min", np.min)

    archive = tools.ParetoFront() if is_multi_obj else tools.HallOfFame(1)
    if is_multi_obj: print("Using tools.ParetoFront for archiving (Greedy Hypervolume selection).")

    # Using eaMuPlusLambda
    final_population, logbook = algorithms.eaMuPlusLambda(
        population, toolbox,
        mu=Config.POPULATION_SIZE, lambda_=Config.OFFSPRING_SIZE,
        cxpb=Config.CX_PROB, mutpb=Config.MUT_PROB, ngen=Config.GENERATIONS,
        stats=stats, halloffame=archive, verbose=True
    )

    print("\n" + "="*50 + "\nOptimization Finished." + "="*50)
    solutions_for_viz = list(archive)
    if is_multi_obj: print(f"Final Pareto Front Size (from archive): {len(solutions_for_viz)}")
    else:
        if solutions_for_viz: print(f"Best Solution Fitness: {solutions_for_viz[0].fitness.values}")
    # print(logbook)

    visualize_results(solutions_for_viz, dist_matrix, edge_comfort_scores,
                      specific_target_nodes, Config.START_NODE)

if __name__ == "__main__":
    main()