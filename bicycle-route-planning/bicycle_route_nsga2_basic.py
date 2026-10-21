import numpy as np
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from functools import partial

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
def visualize_results(hof, dist_matrix, comfort_matrix, target_nodes, start_node):
    """Visualize results"""
    if len(hof[0].fitness.weights) > 1:
        # Multi-objective results
        distances = [ind.fitness.values[0] for ind in hof]
        comforts = [ind.fitness.values[1] for ind in hof]

        plt.figure(figsize=(12, 5))

        # Plot Pareto front
        plt.subplot(1, 2, 1)
        plt.scatter(distances, comforts, c='blue', alpha=0.7)
        plt.xlabel("Total Distance")
        plt.ylabel("Average Comfort")
        plt.title("Pareto Front")
        plt.grid(True)

        # Select best compromise solution (closest to ideal point)
        normalized_dist = (np.array(distances) - min(distances)) / (max(distances) - min(distances))
        normalized_comfort = (np.array(comforts) - min(comforts)) / (max(comforts) - min(comforts))
        scores = np.sqrt(normalized_dist**2 + (1 - normalized_comfort)**2)
        best_idx = np.argmin(scores)
        best_individual = hof[best_idx]

        plt.scatter(distances[best_idx], comforts[best_idx], c='red', s=100, label='Best Compromise')
        plt.legend()
    else:
        # Single-objective results
        best_individual = hof[0]

    # Plot best route
    plt.subplot(1, 2, 2)
    actual_route = [start_node] + [target_nodes[i] for i in best_individual] + [start_node]

    # Generate node coordinates (randomly)
    np.random.seed(42)
    node_positions = np.random.rand(Config.NUM_NODES, 2) * 100

    # Plot all nodes
    plt.scatter(node_positions[:, 0], node_positions[:, 1], c='gray', s=100)

    # Mark start node and target nodes
    plt.scatter(node_positions[start_node, 0], node_positions[start_node, 1],
                c='green', s=200, marker='*', label='Start')

    for node in target_nodes:
        plt.scatter(node_positions[node, 0], node_positions[node, 1],
                    c='blue', s=150, marker='s', label='Target' if node == target_nodes[0] else "")

    # Draw route lines
    for i in range(len(actual_route) - 1):
        start_pos = node_positions[actual_route[i]]
        end_pos = node_positions[actual_route[i + 1]]
        comfort = comfort_matrix[actual_route[i], actual_route[i + 1]]
        color = 'green' if comfort >= 16 else ('orange' if comfort >= 12 else 'red')

        plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                 color=color, linewidth=2, alpha=0.7)

        mid_point = [(start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2]
        plt.text(mid_point[0], mid_point[1], f"{dist_matrix[actual_route[i], actual_route[i + 1]]}",
                 fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    # Calculate real display fitness values
    total_distance = sum(dist_matrix[actual_route[i], actual_route[i+1]] for i in range(len(actual_route)-1))
    avg_comfort = sum(comfort_matrix[actual_route[i], actual_route[i+1]] for i in range(len(actual_route)-1)) / (len(actual_route)-1)

    plt.title(f"Best Route (Distance: {total_distance:.1f}, Avg Comfort: {avg_comfort:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("route_planning_result.png")
    plt.show()

    # Print details
    print("\n" + "=" * 50)
    print("Best Route Details:")
    print(f"Complete Route: {' → '.join(map(str, actual_route))}")

    print("\nSegment Details:")
    total_distance = 0
    total_comfort = 0
    for i in range(len(actual_route) - 1):
        from_node = actual_route[i]
        to_node = actual_route[i + 1]
        dist = dist_matrix[from_node, to_node]
        comfort = comfort_matrix[from_node, to_node]
        total_distance += dist
        total_comfort += comfort

        comfort_status = ""
        if Config.COMFORT_HANDLING in ['constraint', 'both']:
            if comfort < Config.MIN_EDGE_COMFORT:
                comfort_status = " (Below minimum comfort requirement!)"
            elif i == len(actual_route) - 2 and total_comfort / (len(actual_route) - 1) < Config.MIN_AVG_COMFORT:
                comfort_status = " (Average comfort not meeting requirement!)"

        print(f"Node {from_node} → Node {to_node}: Distance={dist}, Comfort={comfort}{comfort_status}")

    avg_comfort = total_comfort / (len(actual_route) - 1)
    print("\nRoute Summary:")
    print(f"Total Distance: {total_distance}")
    print(f"Average Comfort: {avg_comfort:.2f}")

    if Config.COMFORT_HANDLING in ['constraint', 'both']:
        if any(comfort_matrix[actual_route[i], actual_route[i+1]] < Config.MIN_EDGE_COMFORT for i in range(len(actual_route)-1)):
            print("⚠️ Warning: Route contains segments with insufficient comfort!")
        if avg_comfort < Config.MIN_AVG_COMFORT:
            print("⚠️ Warning: Average comfort does not meet the requirement!")
    print("=" * 50)

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