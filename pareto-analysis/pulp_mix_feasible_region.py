import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap # Import this
import itertools

# --- Adjust Hatch Linewidth (Optional) ---
plt.rcParams['hatch.linewidth'] = 1.2

# --- Problem Definition ---
# Constraints dictionary (using single-character hatches)
constraints = {
    'C2_Fiber':    {'a': 0.5, 'b': 1.0, 'c': 12.0, 'label': r'$0.5x_1 + x_2 = 12$', 'hatch_label': 'Violates Fiber', 'color': 'blue', 'hatch': '/'},
    'C3_Sulfur':   {'a': 2.0, 'b': 1.0, 'c': 14.0, 'label': r'$2x_1 + x_2 = 14$', 'hatch_label': 'Violates Sulfur', 'color': 'green', 'hatch': '\\'},
    'C4_Total':    {'a': 1.0, 'b': 1.0, 'c': 15.0, 'label': r'$x_1 + x_2 = 15$', 'hatch_label': 'Violates Total', 'color': 'red', 'hatch': '|'},
    'C5_NonCell':  {'a': 0.5, 'b': 2.5, 'c': 15.0, 'label': r'$0.5x_1 + 2.5x_2 = 15$', 'hatch_label': 'Violates Non-Cell.', 'color': 'purple', 'hatch': '-'},
    'N1_x1_nonneg':{'a': -1.0,'b': 0.0, 'c': 0.0,  'label': r'$x_1 = 0$', 'hatch_label': 'Violates x1>=0', 'color': 'darkgrey', 'hatch': 'x'}, # Made grey darker
    'N2_x2_nonneg':{'a': 0.0, 'b': -1.0,'c': 0.0,  'label': r'$x_2 = 0$', 'hatch_label': 'Violates x2>=0', 'color': 'darkgrey', 'hatch': '+'}  # Made grey darker
}
constraint_keys = list(constraints.keys())

# Define objective function
def objective_func(x1, x2):
    return 4*x1 + 3*x2

# --- Helper Functions (same as before) ---
def find_intersection(key1, key2):
    c1 = constraints[key1]
    c2 = constraints[key2]
    A = np.array([[c1['a'], c1['b']], [c2['a'], c2['b']]])
    b = np.array([c1['c'], c2['c']])
    try:
        intersection = np.linalg.solve(A, b)
        return tuple(intersection)
    except np.linalg.LinAlgError:
        return None

def is_feasible(point, tolerance=1e-9):
    x1, x2 = point
    for data in constraints.values():
        if data['a']*x1 + data['b']*x2 > data['c'] + tolerance:
            return False
    return True

# --- Calculations (same as before) ---
intersections = []
for key1, key2 in itertools.combinations(constraint_keys, 2):
    point = find_intersection(key1, key2)
    if point is not None:
        intersections.append(point)

feasible_vertices = []
for point in intersections:
    if is_feasible(point):
        is_close = False
        for vp in feasible_vertices:
            if np.allclose(point, vp, atol=1e-8):
                is_close = True
                break
        if not is_close:
             feasible_vertices.append(point)

feasible_vertices = [p for p in feasible_vertices if p[0] >= -1e-9 and p[1] >= -1e-9]
feasible_vertices.sort()

profits = {}
for vertex in feasible_vertices:
    profits[vertex] = objective_func(vertex[0], vertex[1])

if not profits:
    print("Error: No feasible region found.")
    exit()

optimal_vertex = max(profits, key=profits.get)
max_profit = profits[optimal_vertex]

# --- Plotting ---

# 1. Define plot ranges and grid
x1_min_plot, x1_max_plot = 0, 25
x2_min_plot, x2_max_plot = 0, 18
x1_range = np.linspace(x1_min_plot, x1_max_plot, 400)
x2_range = np.linspace(x2_min_plot, x2_max_plot, 400)
x1_plot, x2_plot = np.meshgrid(x1_range, x2_range)

# 2. Create the plot
plt.figure(figsize=(10, 8))
ax = plt.gca() # Get current axes

# 3. Shade the INFEASIBLE Regions with COLOR and HATCHES
legend_proxies = []
base_alpha = 0.25 # Base transparency for the colored hatches

for data in constraints.values():
    condition = data['a']*x1_plot + data['b']*x2_plot > data['c'] # Where constraint is violated
    masked_condition = np.where(condition, 1.0, 0.0) # Use 0 for feasible now

    # Create a colormap: transparent for feasible (0), constraint color for infeasible (1)
    cmap = ListedColormap(['none', data['color']]) # 'none' makes level 0 transparent

    # Use contourf to fill with color and apply hatch
    # Levels [0.5, 1.5] target the area where masked_condition is 1.0
    cs = ax.contourf(x1_plot, x2_plot, masked_condition, levels=[0.5, 1.5],
                     cmap=cmap, hatches=[data['hatch']], alpha=base_alpha)

    # Create proxy for legend
    proxy = Patch(facecolor=data['color'], alpha=base_alpha, hatch=data['hatch'], label=data['hatch_label'])
    legend_proxies.append(proxy)

# 4. Plot the Constraint Boundary Lines (thicker, on top)
x1_lines = np.linspace(0, x1_max_plot, 400)
for data in constraints.values():
    a, b, c = data['a'], data['b'], data['c']
    color = data['color']
    label = data['label']
    linewidth = 2.0 # Make lines thicker
    zorder = 5 # Ensure lines are drawn above hatches

    if abs(b) < 1e-9:
        if abs(a) > 1e-9:
             plt.axvline(c/a, color=color, linewidth=linewidth, zorder=zorder, label=label if c/a==0 else None)
    elif abs(a) < 1e-9:
        if abs(b) > 1e-9:
             plt.axhline(c/b, color=color, linewidth=linewidth, zorder=zorder, label=label if c/b==0 else None)
    else:
        x2_lines = (c - a * x1_lines) / b
        mask = (x1_lines >= -1e-9) & (x2_lines >= -1e-9) & (x2_lines < x2_max_plot + 1) & (x1_lines < x1_max_plot+1)
        plt.plot(x1_lines[mask], x2_lines[mask], label=label, color=color, linewidth=linewidth, zorder=zorder)

# 5. Plot Feasible Vertices and Optimal Point (same as before, ensure high zorder)
vertices_x = [v[0] for v in feasible_vertices]
vertices_y = [v[1] for v in feasible_vertices]
plt.scatter(vertices_x, vertices_y, color='black', zorder=7, s=70, label='Feasible Vertices') # Make vertices stand out

plt.scatter([optimal_vertex[0]], [optimal_vertex[1]], color='yellow', s=200, zorder=8, # Increase optimal size/contrast
            edgecolor='black', label=f'Optimal Point (Maximizer)', marker='.') # Use star marker
plt.annotate(f'Maximizer:\n($x_1={optimal_vertex[0]:.2f}, x_2={optimal_vertex[1]:.2f}$)\nMax Profit = {max_profit:.2f}',
             xy=optimal_vertex,
             xytext=(optimal_vertex[0] + 1, optimal_vertex[1] + 1.5), # Adjust text position
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=9, zorder=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.9))

# 6. Create Legend (same as before)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles + legend_proxies, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')


# 7. Final Plot Adjustments (same as before)
plt.xlabel('Amount of Recycled Pulp ($x_1$, kg)')
plt.ylabel('Amount of Wood Pulp ($x_2$, kg)')
plt.title('Constraint Boundaries and Infeasible Region (Colored Hatches)')
plt.xlim(x1_min_plot, x1_max_plot)
plt.ylim(x2_min_plot, x2_max_plot)
plt.grid(True, linestyle=':', alpha=0.6, zorder=1) # Ensure grid is behind everything
plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# --- Reset rcParams if changed ---
# plt.rcParams['hatch.linewidth'] = 1.0

# --- Output Results (same as before) ---
print(f"Constraint Boundaries are drawn on the graph.")
print(f"The infeasible region (violating at least one constraint) is indicated by colored hatches.")
print(f"The clear (unhatched) region is the feasible region.")
print("\nFeasible Vertices Found:")
for i, v in enumerate(feasible_vertices):
    print(f"  V{i}: ({v[0]:.4f}, {v[1]:.4f}), Profit = {profits[v]:.4f}")
print(f"\nMaximizer (Optimal Vertex):")
print(f"  ({optimal_vertex[0]:.4f}, {optimal_vertex[1]:.4f})")
print(f"Maximum Profit: {max_profit:.4f}")



