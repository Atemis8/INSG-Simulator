import numpy as np
import matplotlib.pyplot as plt

L = 1.0
N = 100
x_grid = np.linspace(0, L, N, endpoint=False)  # Uniform grid

xfish = -0.2
Lfish = 0.3

def wrap(x, L):
    return x % L

def is_inside_fish(x, xfish, Lfish, L):
    xstart = wrap(xfish, L)
    xend = wrap(xfish + Lfish, L)
    return (xstart < xend and xstart <= x <= xend) or (xstart >= xend and (x >= xstart or x <= xend))

print(-5 % 3)
# Mask where fish exists
mask = np.array([is_inside_fish(x, xfish, Lfish, L) for x in x_grid], dtype=int)

# Plot for visualization
plt.figure(figsize=(8, 2))
plt.plot(x_grid, mask, 'bo', label="Fish Body Mask")
plt.title(f"Fish from x={xfish:.2f} to x={xfish + Lfish:.2f} (wrapped)")
plt.xlabel("x")
plt.ylabel("Inside Fish (1=yes)")
plt.grid(True)
plt.legend()
plt.show()
