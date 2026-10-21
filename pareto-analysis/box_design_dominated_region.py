import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Constants and Pareto front calculation (same as before)
sqrt2 = np.sqrt(2)
factor = 2 + sqrt2

def V(a, h): return 0.5 * a**2 * h
def S(a, h): return (2 + sqrt2) * a * h + 0.5 * a**2

a_pareto = np.linspace(0.5, 4, 100)
h_pareto = a_pareto / factor
V_pareto = V(a_pareto, h_pareto)
S_pareto = S(a_pareto, h_pareto)

# Create dominance boundary (offset from Pareto front)
offset = 2.0  # Controls how far the boundary extends
V_boundary = np.linspace(min(V_pareto), max(V_pareto), 50)
S_boundary = interp1d(V_pareto, S_pareto, kind='linear')(V_boundary) + offset

# Generate test points
np.random.seed(42)
test_V = np.random.uniform(min(V_pareto), max(V_pareto), 30)
test_S = np.random.uniform(min(S_pareto), max(S_pareto)+5, 30)

# Plotting
plt.figure(figsize=(10, 6))

# Plot Pareto front
plt.plot(V_pareto, S_pareto, 'b-', linewidth=3, label='Pareto front')

# Plot dominance boundary
plt.plot(V_boundary, S_boundary, 'r--', linewidth=2, label='Dominance boundary')

# Fill dominated region
plt.fill_between(V_boundary, S_pareto[:50], S_boundary, 
                 color='red', alpha=0.15, label='Dominated region')

# Classify and plot test points
for v, s in zip(test_V, test_S):
    # Find closest Pareto point
    idx = np.argmin(np.abs(V_pareto - v))
    if s > S_pareto[idx]:
        plt.scatter(v, s, color='red', s=50, alpha=0.7)
    else:
        plt.scatter(v, s, color='green', s=50, alpha=0.7)

plt.xlabel('Volume (V)')
plt.ylabel('Surface Area (S)')
plt.title('Dominated Region Boundary with Test Points')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()