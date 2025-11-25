import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def comp_field(field, exc):
    fig = plt.figure(figsize=(12, 6))

    # 2D plot
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(field, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax1)
    ax1.set_title("2D Field Error (Numerical)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2 = fig.add_subplot(122)
    im = ax2.imshow(exc, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax2)
    ax2.set_title("2D Field Error (Exact)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    
    plt.tight_layout()
    plt.show()

def plot_field(fig, field, index, num):

    x = np.linspace(0, 2 * np.pi, field.shape[1])
    y = np.linspace(0, 2 * np.pi, field.shape[0])

    # 2D plot
    ax1 = fig.add_subplot(1, num, index)
    im = ax1.imshow(field, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax1)
    ax1.set_title("2D Field Error (Numerical - Exact)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    
    plt.tight_layout()

def compare_field():

    # Load numerical results
    divu_num = np.load("divu.npy")
    divv_num = np.load("divv.npy")

    # Grid setup
    ny, nx = divu_num.shape  # assumes same shape for both fields
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    X, Y = np.meshgrid(x, y)

    # Compute exact expressions

    # For divu = d(uu)/dx + d(uv)/dy
    exact_divu = (
        np.sin(2 * X) * np.cos(Y)**2
        - 0.5 * np.sin(2 * X) * np.cos(2 * Y)
    )

    # For divv = d(uv)/dx + d(vv)/dy
    exact_divv = (
        (np.cos(X) ** 2 - 0.5 * np.cos(2 * X)) * np.sin(2 * Y)
    )

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # divu comparison
    im0 = axes[0, 0].imshow(divu_num, origin="lower", cmap='viridis')
    axes[0, 0].set_title("Numerical divu")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(exact_divu - divu_num, origin="lower", cmap='coolwarm')
    axes[0, 1].set_title("Exact divu")
    plt.colorbar(im1, ax=axes[0, 1])

    # divv comparison
    im2 = axes[1, 0].imshow(divv_num, origin="lower", cmap='viridis')
    axes[1, 0].set_title("Numerical divv")
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(exact_divv - divv_num, origin="lower", cmap='coolwarm')
    axes[1, 1].set_title("Exact divv")
    plt.colorbar(im3, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()


anal = np.load("anal.npy")
num = np.load("num.npy")

print(np.max(num), np.min(num))

fig = plt.figure(figsize=(12, 6))
plot_field(fig, anal, 1, 2)
plot_field(fig, num, 2, 2)
plt.show()

