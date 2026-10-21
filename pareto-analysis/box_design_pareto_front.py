import numpy as np
import matplotlib.pyplot as plt

# Constants
sqrt2 = np.sqrt(2)
factor = 2 + sqrt2  # 2 + √2

# Define functions
def V(a, h):
    return 0.5 * np.array(a)**2 * np.array(h)

def S(a, h):
    return 2*np.array(a)*np.array(h) + sqrt2*np.array(a)*np.array(h) + 0.5*np.array(a)**2

# Create grid
a = np.linspace(0.1, 5, 100)
h = np.linspace(0.1, 3, 100)
A, H = np.meshgrid(a, h)

# Calculate V and S
V_vals = V(A, H)
S_vals = S(A, H)

# Efficient set (h = a/(2 + √2))
a_eff = np.linspace(0.1, 5, 50)
h_eff = a_eff / factor

# Create figure
plt.figure(figsize=(12, 6))

# =============================================
# 1. Level Curves Plot with Efficient Set
# =============================================
plt.subplot(1, 2, 1)

# Plot level curves for V (blue)
CS1 = plt.contour(A, H, V_vals, levels=[0.5, 1, 2, 3], colors='blue', alpha=0.7)
plt.clabel(CS1, inline=True, fontsize=9, fmt='V=%1.1f')
# Create proxy artist for legend
V_legend = plt.Line2D([0], [0], color='blue', label='Volume (V) level curves')

# Plot level curves for S (red)
CS2 = plt.contour(A, H, S_vals, levels=[2, 4, 6, 8], colors='red', alpha=0.7)
plt.clabel(CS2, inline=True, fontsize=9, fmt='S=%1.1f')
S_legend = plt.Line2D([0], [0], color='red', label='Surface (S) level curves')

# Plot efficient set
eff_line, = plt.plot(a_eff, h_eff, 'k--', linewidth=2, label='Efficient set')

# Mark Pareto optimal points
pareto_points = plt.scatter([1, 2, 3], [1/factor, 2/factor, 3/factor], 
                           color='lime', s=100, edgecolor='black', label='Pareto points')

plt.xlabel('a (base length)')
plt.ylabel('h (height)')
plt.title('Level Curves and Efficient Set')
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(handles=[V_legend, S_legend, eff_line, pareto_points])

# =============================================
# 2. Pareto Front Plot
# =============================================
plt.subplot(1, 2, 2)

# Calculate Pareto front
V_pareto = V(a_eff, h_eff)
S_pareto = S(a_eff, h_eff)


# Plot Pareto front
pareto_curve, = plt.plot(V_pareto, S_pareto, 'b-', linewidth=2, label='Pareto front')

# Mark corresponding points
plt.scatter(V([1,2,3], [1/factor,2/factor,3/factor]), 
            S([1,2,3], [1/factor,2/factor,3/factor]),
            color='lime', s=100, edgecolor='black')

plt.xlabel('Volume (V)')
plt.ylabel('Surface Area (S)')
plt.title('Pareto Front: Optimal Trade-off')
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()