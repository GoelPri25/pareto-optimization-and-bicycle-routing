# Import the PuLP library
import pulp

# Define the problem: Maximize Profit
prob = pulp.LpProblem("PaperPulpMix", pulp.LpMaximize) # Changed name and type

# Define the decision variables: x1 and x2 represent kg amounts
# Use lowBound=0 for non-negativity. Default category is Continuous.
x1 = pulp.LpVariable("RecycledPulp_x1", lowBound=0) # Renamed variable
x2 = pulp.LpVariable("WoodPulp_x2", lowBound=0)    # Renamed variable

# Define the objective function: Maximize P = 4*x1 + 3*x2
prob += 4 * x1 + 3 * x2, "TotalProfit" # Updated objective function

# Define the constraints
# Constraints of the pulp-mixing problem
prob += 0.5*x1 + 1.0*x2 <= 12.0, "Fiber_Limit"        # Constraint (2)
prob += 2.0*x1 + 1.0*x2 <= 14.0, "Sulfur_Limit"       # Constraint (3)
prob += 1.0*x1 + 1.0*x2 <= 15.0, "Total_Weight_Limit" # Constraint (4)
prob += 0.5*x1 + 2.5*x2 <= 15.0, "NonCellulose_Limit" # Constraint (5)
# Constraints x1 >= 0 and x2 >= 0 are handled by lowBound=0

# Solve the problem
prob.solve()

# Print the status of the problem
print("Status: ", pulp.LpStatus[prob.status])

# Check if the solution is optimal before printing results
if prob.status == pulp.LpStatusOptimal:
    # Print the optimal values of the decision variables
    # Using .value() or .varValue is standard
    print(f"x1 (Recycled Pulp) = {x1.value():.4f} kg")
    print(f"x2 (Wood Pulp)     = {x2.value():.4f} kg")
    # Calculate x3 for informational purposes
    x3_val = 15.0 - x1.value() - x2.value()
    print(f"x3 (Pure Cellulose)= {x3_val:.4f} kg") # Calculated amount

    # Print the optimal objective value
    # Using .value() or pulp.value() is standard
    print(f"Optimal Profit Value = {prob.objective.value():.4f}")
else:
    print("Solver did not find an optimal solution.")
    print(f"Status code: {prob.status}")