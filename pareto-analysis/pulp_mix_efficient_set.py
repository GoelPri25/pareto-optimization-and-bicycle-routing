import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import itertools

# --- Adjust Hatch Linewidth (Optional) ---
plt.rcParams['hatch.linewidth'] = 1.0

# --- Problem Definition & Helpers ---
# Constraint definitions
constraints = {
    'C2_Fiber':    {'a': 0.5, 'b': 1.0, 'c': 12.0, 'label': r'$0.5x_1 + x_2 \leq 12$', 'hatch_label': 'Violates Fiber', 'color': 'blue', 'hatch': '/'},
    'C3_Sulfur':   {'a': 2.0, 'b': 1.0, 'c': 14.0, 'label': r'$2x_1 + x_2 \leq 14$', 'hatch_label': 'Violates Sulfur', 'color': 'green', 'hatch': '\\'},
    'C4_Total':    {'a': 1.0, 'b': 1.0, 'c': 15.0, 'label': r'$x_1 + x_2 \leq 15$', 'hatch_label': 'Violates Total', 'color': 'red', 'hatch': '|'},
    'C5_NonCell':  {'a': 0.5, 'b': 2.5, 'c': 15.0, 'label': r'$0.5x_1 + 2.5x_2 \leq 15$', 'hatch_label': 'Violates Non-Cellulose', 'color': 'purple', 'hatch': '-'}, # Label fixed
    # Using standard form a*x <= c equivalent for non-negativity checks in is_feasible
    'N1_x1_nonneg':{'a': -1.0,'b': 0.0, 'c': 0.0,  'label': r'$x_1 \geq 0$', 'hatch_label': 'Violates $x_1 \geq 0$', 'color': 'black', 'hatch': 'x'},
    'N2_x2_nonneg':{'a': 0.0, 'b': -1.0,'c': 0.0,  'label': r'$x_2 \geq 0$', 'hatch_label': 'Violates $x_2 \geq 0$', 'color': 'black', 'hatch': '+'}
}
constraint_keys = list(constraints.keys())

def objective_P(x1, x2): return 4*x1 + 3*x2
def objective_R(x1, x2): return x1

def find_intersection(key1, key2):
    c1, c2 = constraints[key1], constraints[key2]
    # Adjust for standard form representation if needed for calculation
    A = np.array([[-c1['a'], -c1['b']] if key1.startswith('N') else [c1['a'], c1['b']],
                  [-c2['a'], -c2['b']] if key2.startswith('N') else [c2['a'], c2['b']]])
    b = np.array([-c1['c'] if key1.startswith('N') else c1['c'],
                  -c2['c'] if key2.startswith('N') else c2['c']])
    # Let's use the direct definition for solving intersections based on boundary = c
    A = np.array([[constraints[key1]['a'], constraints[key1]['b']],
                  [constraints[key2]['a'], constraints[key2]['b']]])
    b = np.array([constraints[key1]['c'], constraints[key2]['c']])

    try: return tuple(np.linalg.solve(A, b))
    except np.linalg.LinAlgError: return None

def is_feasible(point, tolerance=1e-9):
    if point is None: return False
    x1, x2 = point
    # Check standard form constraints a*x <= c
    for key, data in constraints.items():
         val = data['a']*x1 + data['b']*x2
         if val > data['c'] + tolerance:
             return False
    # Also explicit non-negativity check
    if x1 < -tolerance or x2 < -tolerance:
        return False
    return True


# --- Calculations (find feasible vertices) ---
intersections = []
# Intersections with axes (x1=0 or x2=0)
for key in constraint_keys:
    if key != 'N1_x1_nonneg': # Intersect with x2=0 axis (key N2)
         pt = find_intersection(key, 'N2_x2_nonneg')
         if pt: intersections.append(pt)
    if key != 'N2_x2_nonneg': # Intersect with x1=0 axis (key N1)
         pt = find_intersection(key, 'N1_x1_nonneg')
         if pt: intersections.append(pt)
# Intersections between non-axis constraints
non_axis_keys = [k for k in constraint_keys if not k.startswith('N')]
for k1, k2 in itertools.combinations(non_axis_keys, 2):
    pt = find_intersection(k1, k2)
    if pt: intersections.append(pt)
intersections.append((0.0, 0.0)) # Add origin

# Filter for feasibility and uniqueness
feasible_vertices_set = set()
tolerance_check = 1e-8
for p in intersections:
    # Round slightly before feasibility check for robustness near boundaries
    p_rounded = (round(p[0], 8), round(p[1], 8))
    if is_feasible(p_rounded, tolerance=tolerance_check):
        feasible_vertices_set.add(p_rounded)

feasible_vertices = sorted(list(feasible_vertices_set), key=lambda k: (k[0], k[1]))

if not feasible_vertices:
    print("Error: No feasible vertices found!")
    exit()

# Calculate objective ranges over feasible vertices
min_P_feas = min(objective_P(v[0], v[1]) for v in feasible_vertices)
max_P_feas = max(objective_P(v[0], v[1]) for v in feasible_vertices)
min_R_feas = min(objective_R(v[0], v[1]) for v in feasible_vertices)
max_R_feas = max(objective_R(v[0], v[1]) for v in feasible_vertices)

vertex_max_R = max(feasible_vertices, key=lambda v: objective_R(v[0], v[1]))
vertex_max_P = max(feasible_vertices, key=lambda v: objective_P(v[0], v[1]))

# Define test points
point_A = (4, 4) # Sample inefficient point
point_B = vertex_max_R # Max E vertex
point_C = vertex_max_P # Max P vertex

# Check if key vertices were found (redundant check if feasible_vertices is not empty)
if point_B is None or point_C is None:
    print("Error: Could not determine key vertices.")
    exit()

points_to_show = {'A (Inefficient Sample)': point_A, 'B (Max Recycled)': point_B, 'C (Max Profit)': point_C}
point_colors = {'A (Inefficient Sample)': 'orange', 'B (Max Recycled)': 'deepskyblue', 'C (Max Profit)': 'magenta'}
point_styles_P = {'A (Inefficient Sample)': '--', 'B (Max Recycled)': '--', 'C (Max Profit)': '--'} # Make specific P lines dashed
point_styles_E = {'A (Inefficient Sample)': '-.', 'B (Max Recycled)': '-.', 'C (Max Profit)': '-.'} # Make specific E lines dash-dot
point_sizes = {'A (Inefficient Sample)': 80, 'B (Max Recycled)': 120, 'C (Max Profit)': 200}
point_markers = {'A (Inefficient Sample)': 'o', 'B (Max Recycled)': 's', 'C (Max Profit)': '*'}


# --- Plotting ---
x1_min_plot, x1_max_plot = -1, 16
x2_min_plot, x2_max_plot = -1, 16
x1_range = np.linspace(x1_min_plot, x1_max_plot, 400)
x2_range = np.linspace(x2_min_plot, x2_max_plot, 400)
x1_plot, x2_plot = np.meshgrid(x1_range, x2_range)

plt.figure(figsize=(10, 8)) # Adjusted size
ax = plt.gca()
legend_items = {} # Use dictionary for reliable legend creation

# 1. Shade the INFEASIBLE Regions
base_alpha = 0.15
for key, data in constraints.items():
    # Logic for determining infeasible area based on original constraint intent
    a, b, c = data['a'], data['b'], data['c']
    if key.startswith('N'): # Non-negativity
        if key == 'N1_x1_nonneg': condition = x1_plot < 0
        else: condition = x2_plot < 0
    else: # Other constraints (assumed <=)
        condition = a*x1_plot + b*x2_plot > c

    masked_condition = np.where(condition, 1.0, 0.0)
    cmap = ListedColormap(['none', data['color']])
    ax.contourf(x1_plot, x2_plot, masked_condition, levels=[0.5, 1.5],
                cmap=cmap, hatches=[data['hatch']], alpha=base_alpha, zorder=2) # zorder for hatches

    # Add hatch proxy if label not already in legend
    label = data['hatch_label']
    if label not in legend_items:
         proxy = Patch(facecolor=data['color'], alpha=base_alpha, hatch=data['hatch'])
         legend_items[label] = proxy


# --- ADD GENERAL LEVEL SETS (low zorder) ---
num_general_levels = 8 # More levels for general overview
general_level_alpha = 0.5
general_level_zorder = 4 # Below constraints and specific items

# General Profit Levels
P_general_values = np.linspace(min_P_feas, max_P_feas, num_general_levels)
x1_level_lines = np.linspace(x1_min_plot, x1_max_plot, 200)
gen_P_label_added = False
for k_P in P_general_values:
    x2_level_P = (k_P - 4 * x1_level_lines) / 3
    mask_P = (x1_level_lines >= x1_min_plot) & (x1_level_lines <= x1_max_plot) & \
             (x2_level_P >= x2_min_plot) & (x2_level_P <= x2_max_plot)
    handle_gen_P, = ax.plot(x1_level_lines[mask_P], x2_level_P[mask_P],
                           color='grey', linestyle=':', linewidth=0.8,
                           alpha=general_level_alpha, zorder=general_level_zorder)
    if not gen_P_label_added:
        legend_items[f"General Profit Levels ({num_general_levels})"] = handle_gen_P
        gen_P_label_added = True

# General Recycled Levels (x1=k)
E_general_values = np.linspace(min_R_feas, max_R_feas, num_general_levels)
y_plot_min_gen, y_plot_max_gen = ax.get_ylim() # Get current y limits
ymin_rel_gen = (max(0, y_plot_min_gen) - y_plot_min_gen) / (y_plot_max_gen - y_plot_min_gen) if (y_plot_max_gen - y_plot_min_gen) else 0
ymax_rel_gen = (min(x2_max_plot, y_plot_max_gen) - y_plot_min_gen) / (y_plot_max_gen - y_plot_min_gen) if (y_plot_max_gen - y_plot_min_gen) else 1
gen_E_label_added = False
for k_E in E_general_values:
    handle_gen_E = ax.axvline(k_E, ymin=ymin_rel_gen, ymax=ymax_rel_gen,
                             color='darkcyan', linestyle=':', linewidth=0.8, # Changed color slightly
                             alpha=general_level_alpha, zorder=general_level_zorder)
    if not gen_E_label_added:
        legend_items[f"General Recycled Levels ({num_general_levels})"] = handle_gen_E
        gen_E_label_added = True
# --- END GENERAL LEVEL SETS ---


# 2. Plot the Constraint Boundary Lines (higher zorder)
x1_lines = np.linspace(0, x1_max_plot, 400)
for key, data in constraints.items(): # Use original labels for lines
    a, b, c = data['a'], data['b'], data['c']
    color = data['color']
    label = data['label'].replace('=','\leq') # Display as inequality boundary
    linewidth = 1.5
    zorder = 5 # Above general level sets
    handle = None
    # Plot axis lines explicitly black if desired
    line_color = 'black' if key.startswith('N') else color

    if key == 'N1_x1_nonneg': # y-axis (x1=0)
        handle = plt.axvline(0, color=line_color, linewidth=linewidth, zorder=zorder)
    elif key == 'N2_x2_nonneg': # x-axis (x2=0)
        handle = plt.axhline(0, color=line_color, linewidth=linewidth, zorder=zorder)
    elif abs(b) < 1e-9: # Vertical line
        if abs(a) > 1e-9:
            x_val = c/a
            if x_val >= 0 : handle = plt.axvline(x_val, color=line_color, linewidth=linewidth, zorder=zorder)
    elif abs(a) < 1e-9: # Horizontal line
        if abs(b) > 1e-9:
            y_val = c/b
            if y_val >= 0: handle = plt.axhline(y_val, color=line_color, linewidth=linewidth, zorder=zorder)
    else: # Slanted line
        x2_lines = (c - a * x1_lines) / b
        mask = (x1_lines >= -1e-9) & (x2_lines >= -1e-9) & (x2_lines < x2_max_plot + 1) & (x1_lines < x1_max_plot+1)
        if np.any(mask): handle, = plt.plot(x1_lines[mask], x2_lines[mask], color=line_color, linewidth=linewidth, zorder=zorder)

    if handle and label not in legend_items:
        legend_items[label] = handle


# 3. Plot Specific Level Sets for points A, B, C (higher zorder)
x1_level_lines_specific = np.linspace(x1_min_plot, x1_max_plot, 200)
specific_level_zorder = 7 # Above constraint lines, below points

for name, point in points_to_show.items():
    px, py = point
    color = point_colors[name]
    style_P = point_styles_P[name] # Dashed for P
    style_E = point_styles_E[name] # Dash-dot for E

    # Specific Level set for P = 4x1 + 3x2 = P(point)
    P_val = objective_P(px, py)
    x2_level_P = (P_val - 4 * x1_level_lines_specific) / 3
    mask_P = (x1_level_lines_specific >= x1_min_plot) & (x1_level_lines_specific <= x1_max_plot) & \
             (x2_level_P >= x2_min_plot) & (x2_level_P <= x2_max_plot)
    h_P_spec, = ax.plot(x1_level_lines_specific[mask_P], x2_level_P[mask_P],
                        color=color, linestyle=style_P, linewidth=1.5, zorder=specific_level_zorder)
    label_P_spec = f"Profit Level (Point {name[0]})" # Add point marker to label
    if label_P_spec not in legend_items: legend_items[label_P_spec] = h_P_spec


    # Specific Level set for E = x1 = E(point)
    E_val = objective_R(px, py)
    y_plot_min_spec, y_plot_max_spec = ax.get_ylim() # Get current y limits
    ymin_rel_spec = (max(0, y_plot_min_spec) - y_plot_min_spec) / (y_plot_max_spec - y_plot_min_spec) if (y_plot_max_spec - y_plot_min_spec) else 0
    ymax_rel_spec = (min(x2_max_plot, y_plot_max_spec) - y_plot_min_spec) / (y_plot_max_spec - y_plot_min_spec) if (y_plot_max_spec - y_plot_min_spec) else 1
    h_E_spec = ax.axvline(E_val, ymin=ymin_rel_spec, ymax=ymax_rel_spec,
                         color=color, linestyle=style_E, linewidth=1.5, zorder=specific_level_zorder)
    label_E_spec = f"Recycled Level (Point {name[0]})"
    if label_E_spec not in legend_items: legend_items[label_E_spec] = h_E_spec


# 4. Plot Efficient Set Boundary (above general levels, below points)
v_e = vertex_max_R
v_p = vertex_max_P
eff_set_x = [v_e[0], v_p[0]]
eff_set_y = [v_e[1], v_p[1]]
h_eff, = ax.plot(eff_set_x, eff_set_y, color='lime', linewidth=4, zorder=6, solid_capstyle='round') # zorder=6
legend_items['Efficient Set'] = h_eff

# 5. Plot the Specific Points A, B, C (highest zorder)
for name, point in points_to_show.items():
    px, py = point
    handle_pt = ax.scatter([px], [py],
                           color=point_colors[name],
                           marker=point_markers[name],
                           s=point_sizes[name],
                           zorder=10, edgecolor='black') # zorder=10 for points
    legend_items[f"Point {name}"] = handle_pt # Add point to legend

    # Optional: Indicate improvement direction/dominance cone for point A
    if name == 'A (Inefficient Sample)': # Adjusted label name
        arr_len = 0.8 # Adjust arrow length/position as needed
        ax.arrow(px, py, arr_len*4/5, arr_len*3/5, head_width=0.3, head_length=0.4, fc='black', ec='black', linestyle='-', zorder=11, length_includes_head=True) # Dir P <4,3> scaled
        ax.arrow(px, py, arr_len, 0.0, head_width=0.3, head_length=0.4, fc='black', ec='black', linestyle='-', zorder=11, length_includes_head=True) # Dir E <1,0> scaled
        # ax.text(px+arr_len+0.1, py+0.1, "Improvement\nDirections", fontsize=8, zorder=11) # Optional text


# 6. Create Legend from collected items
plt.legend(handles=legend_items.values(), labels=legend_items.keys(),
           loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small',
           title="Legend")


# 7. Final Plot Adjustments
plt.xlabel('Amount of Recycled Pulp ($x_1$, kg)')
plt.ylabel('Amount of Wood Pulp ($x_2$, kg)')
plt.title('Level Sets and Efficient Set for Bi-Objective Pulp Mix')
plt.xlim(x1_min_plot, x1_max_plot)
plt.ylim(x2_min_plot, x2_max_plot)
plt.grid(True, linestyle=':', alpha=0.6, zorder=1) # Grid lowest zorder
plt.axhline(0, color='black', linewidth=0.75, zorder=3) # Keep axes visible
plt.axvline(0, color='black', linewidth=0.75, zorder=3)
plt.tight_layout(rect=[0, 0, 0.83, 1]) # Adjust rect for legend width
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# --- Output Results ---
#
print(f"Constraint Boundaries and Level Sets are drawn.")
print(f"The infeasible region is indicated by colored hatches.")
print(f"The clear region is feasible.")
print(f"The thick lime green line is the Efficient Set.")
print("\nFeasible Vertices Found:")
if not feasible_vertices:
     print("  None")
else:
    for i, v in enumerate(feasible_vertices):
        print(f"  V{i}: ({v[0]:.4f}, {v[1]:.4f}), P = {objective_P(v[0],v[1]):.2f}, E = {objective_R(v[0],v[1]):.2f}")

print(f"\nVertex Maximizing E (Recycled) (Point B): ({vertex_max_R[0]:.4f}, {vertex_max_R[1]:.4f})")
print(f"Vertex Maximizing P (Profit)    (Point C): ({vertex_max_P[0]:.4f}, {vertex_max_P[1]:.4f})")
print(f"Sample Inefficient Point          (Point A): ({point_A[0]:.4f}, {point_A[1]:.4f})")