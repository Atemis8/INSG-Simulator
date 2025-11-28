import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap

def load_params(directory):
    param_file = os.path.join(directory, "sim.params")
    parameters = {}

    if not os.path.isfile(param_file):
        raise FileNotFoundError(f"'sim.params' not found in directory: {directory}")

    with open(param_file, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=')
                if '.' in value or 'e' in value.lower(): parameters[key] = float(value)
                else: parameters[key] = int(value)

    return parameters

def load_frame_files(frames_dir, prefix):
    pattern = os.path.join(frames_dir, f"{prefix}_*.npy")
    frame_files = sorted(glob.glob(pattern))
    if not frame_files:
        raise SystemExit(f"No files matching '{pattern}'")
    frame_dict = {}
    for path in frame_files:
        m = re.search(rf"{re.escape(prefix)}_(\d{{4,5}})\.npy$", path)
        if m:
            frame_dict[int(m.group(1))] = path
    available_frames = sorted(frame_dict.keys())
    print(f"Found {len(available_frames)} {prefix} frames: {available_frames[0]} → {available_frames[-1]}")
    return frame_dict, available_frames

def create_animation(frames_dir, data_loader, plot_setup, update_func, fps=24, output_file=None):
    if not os.path.isdir(frames_dir):
        raise SystemExit(f"Error: directory '{frames_dir}' not found.")
    data, fig, ax, im, available_frames = plot_setup()
    ani = animation.FuncAnimation(fig, update_func, frames=len(available_frames), interval=1000/fps, blit=True)
    if output_file:
        writervideo = animation.FFMpegWriter(fps=fps)
        print(f"Saving → {output_file} …")
        ani.save(output_file, writer=writervideo)
        print("Done.")
    else:
        plt.show()
    return ani

def wuv():
    frames_dir = "dump_ctrl2"
    w_prefix = "w"
    m_prefix = "mask"
    output_file = "vorticity.mp4"
    def setup():
        w_frame_dict, available_frames_w = load_frame_files(frames_dir, w_prefix)
        m_frame_dict, available_frames_m = load_frame_files(frames_dir, m_prefix)

        w_data0 = np.load(w_frame_dict[available_frames_w[0]]) 
        m_data0 = np.load(m_frame_dict[available_frames_m[0]])

        ny, nx = w_data0.shape
        fig, ax = plt.subplots(figsize=(8, 8 * ny / nx))

        from matplotlib.colors import ListedColormap
        black_cmap = ListedColormap([[0, 0, 0, 0], [0, 0, 0, 1]])

        w_data = np.load(w_frame_dict[available_frames_w[100]])
        rang = np.max(np.abs(w_data))
        im = ax.imshow(w_data0[:-1, :-1], cmap="coolwarm", origin="lower", animated=True, vmin=-rang, vmax=rang)
        mask = ax.imshow(m_data0, cmap=black_cmap, origin="lower", alpha=1.0, vmin=0, vmax=1, animated=True)

        ax.set_aspect('equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Vorticity")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Vorticity with Velocity Vectors - Frame {available_frames_w[0]}")
        return (w_frame_dict, m_frame_dict, fig, ax, im, mask, available_frames_w, available_frames_m)
    def update(i):
        w_frame_dict, m_frame_dict, _, ax, im, mask, available_frames_w, available_frames_m = setup.data
        frame_num = available_frames_w[i]

        w_arr = np.load(w_frame_dict[frame_num])
        im.set_array(w_arr[:-1, :-1])

        m_arr = np.load(m_frame_dict[frame_num])
        mask.set_array(m_arr)
        ax.set_title(f"Vorticity with Velocity Vectors - Frame {frame_num}")
        return im, mask
    
    setup.data = setup()
    if not os.path.isdir(frames_dir): raise SystemExit(f"Error: directory '{frames_dir}' not found.")
    _, _, fig, _, _, _, available_frames_w, _ = setup.data
    ani = animation.FuncAnimation(fig, update, frames=len(available_frames_w), interval=1000/24, blit=True)
    writervideo = animation.FFMpegWriter(fps=24)
    print(f"Saving → {output_file} …")
    ani.save(output_file, writer=writervideo)
    print("Done.")
    return ani

def wuv_periodic(frames_dir="dump"):
    w_prefix = "w"
    m_prefix = "mask"
    output_file = "vorticity.mp4"
    dupl = (2, 2)
    domain = (1.25, 0.625)

    w_frame_dict, available_frames_w = load_frame_files(frames_dir, w_prefix)
    m_frame_dict, available_frames_m = load_frame_files(frames_dir, m_prefix)

    if not os.path.isdir(frames_dir):
        raise SystemExit(f"Error: directory '{frames_dir}' not found.")

    w_data0 = np.load(w_frame_dict[available_frames_w[0]])
    m_data0 = np.load(m_frame_dict[available_frames_m[0]])
    ny, nx = w_data0.shape
    fig, ax = plt.subplots(figsize=(8, 8 * ny / nx))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('off')

    im = ax.imshow(np.tile(w_data0[:-1, :-1], dupl), cmap="coolwarm", origin="lower",
                   animated=True, vmin=-5.0, vmax=5.0)
    
    black_cmap = ListedColormap([[0, 0, 0, 0], [0, 0, 0, 1]])
    mask = ax.imshow(np.tile(m_data0, dupl), cmap=black_cmap, origin="lower",
                     alpha=1.0, vmin=0, vmax=1, animated=True)

    ax.set_aspect('equal')
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbar = fig.colorbar(im, cax=cax)
    #cbar.set_label("Vorticity")
    #ax.set_xlabel("X")
    #ax.set_ylabel("Y")
    #title = ax.set_title("")

    def update(i):
        frame_num = available_frames_w[i]
        w_arr = np.load(w_frame_dict[frame_num])[:-1, :-1]
        m_arr = np.load(m_frame_dict[frame_num])

        im.set_array(np.tile(w_arr, dupl))
        mask.set_array(np.tile(m_arr, dupl))
        #title.set_text(f"Vorticity with Velocity Vectors - Frame {frame_num}")
        return im, mask

    ani = animation.FuncAnimation(fig, update, frames=len(available_frames_w),
                                  interval=1000/24, blit=True)

    writervideo = animation.FFMpegWriter(fps=24, codec='libx264', extra_args=['-crf', '28', '-preset', 'slow'])
    print(f"Saving → {output_file} …")
    ani.save(output_file, writer=writervideo)
    print("Done.")
    return ani


def pressure_plot_channel(
    frame_index=120,
    physical_domain=(1.25, 0.625),
    display_domain=(2.5, 1.25),
    frames_dir="dump_period",
    w_prefix="P",
    m_prefix="mask",
    tag="period1000"
):
    # Load available frame files
    w_frame_dict, available_frames_w = load_frame_files(frames_dir, w_prefix)
    m_frame_dict, available_frames_m = load_frame_files(frames_dir, m_prefix)

    if not available_frames_w or not available_frames_m:
        raise SystemExit("No frame data found in the specified directory.")
    
    params = load_params(frames_dir)
    Lfish = params["Lfish"]
    L = params["L"]
    H = params["H"]
    
    frame_num = available_frames_w[frame_index]
    w_data = np.load(w_frame_dict[frame_num]) / (rho * Lfish * Lfish / T ** 2)
    m_data = np.load(m_frame_dict[frame_num])

    # Physical size of one periodic tile
    Lx_tile, Ly_tile = physical_domain

    # Display domain size
    Lx_total, Ly_total = display_domain

    # Plotting
    fig, ax = plt.subplots(constrained_layout=False, figsize=(8, 2))
    black_cmap = ListedColormap([[0, 0, 0, 0], [0, 0, 0, 1]])
    extent = [0.0, 1.0, 0.0, 0.25]

    im = ax.imshow(w_data, cmap="coolwarm", origin="lower", vmin=-0.3, vmax=0.3, extent=extent)
    mask = ax.imshow(m_data, cmap=black_cmap, origin="lower", alpha=1.0, vmin=0, vmax=1, extent=extent)

    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    # make the bar narrower: size="3%" instead of 5%
    cax = divider.append_axes("right", size="3%", pad=0.1)

    # 3) actually draw the colorbar, shrinking it a little
    cbar = fig.colorbar(im, cax=cax, shrink=0.85)
    cbar.ax.tick_params(labelsize=8)               # smaller tick labels
    cbar.set_label(r"$\dfrac{P - P_{\rm ref}}{\rho\,L_{\rm fish}^2/T^2}$")

    ax.set_xlabel(r"$L^\ast = \dfrac{L}{L_{fish}}$")
    ax.set_ylabel(r"H$^\ast = \dfrac{H}{L_{fish}}$")
    safe_time = str(time).replace('.', '_')
    plt.savefig(f"Figures/pressure_{tag}_t_{safe_time}.pdf",)
    plt.close()


def wuv_plot_tiled_periodic(
    frame_index=120,
    physical_domain=(1.25, 0.625),
    display_domain=(2.5, 1.25),
    frames_dir="dump_period",
    w_prefix="P",
    m_prefix="mask",
    tag="period1000"
):
    # Load available frame files
    w_frame_dict, available_frames_w = load_frame_files(frames_dir, w_prefix)
    m_frame_dict, available_frames_m = load_frame_files(frames_dir, m_prefix)

    if not available_frames_w or not available_frames_m:
        raise SystemExit("No frame data found in the specified directory.")
    
    params = load_params(frames_dir)
    Lfish = params["Lfish"]
    L = params["L"]
    H = params["H"]
    
    frame_num = available_frames_w[frame_index]
    w_data = np.load(w_frame_dict[frame_num])
    m_data = np.load(m_frame_dict[frame_num])

    # Physical size of one periodic tile
    Lx_tile, Ly_tile = physical_domain

    # Display domain size
    Lx_total, Ly_total = display_domain

    # Compute number of tiles needed in each direction
    ntile_x = int(np.ceil(Lx_total / Lx_tile))
    ntile_y = int(np.ceil(Ly_total / Ly_tile))

    # Tile the data
    w_tiled = np.tile(w_data, (ntile_y, ntile_x))  / (rho * Lfish * Lfish / T ** 2)
    m_tiled = np.tile(m_data, (ntile_y, ntile_x))

    # Compute exact extent in physical coordinates
    extent = [0, ntile_x * Lx_tile, 0, ntile_y * Ly_tile]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8 * w_tiled.shape[0] / w_tiled.shape[1]),constrained_layout=True)
    black_cmap = ListedColormap([[0, 0, 0, 0], [0, 0, 0, 1]])

    im = ax.imshow(w_tiled, cmap="coolwarm", origin="lower", vmin=-0.3, vmax=0.3, extent=extent)
    mask = ax.imshow(m_tiled, cmap=black_cmap, origin="lower", alpha=1.0, vmin=0, vmax=1, extent=extent)

    ax.set_aspect('equal')
    #ivider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbar = fig.colorbar(im, cax=cax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(r" $\dfrac{P-P_{ref}}{\rho L_{fish}^2/T^2}$")
    ax.set_xlabel(r"$L^\ast = \dfrac{L}{L_{fish}}$")
    ax.set_ylabel(r"H$^\ast = \dfrac{H}{L_{fish}}$")
    safe_time = str(time).replace('.', '_')
    plt.savefig(f"Figures/pressure_{tag}_t_{safe_time}.pdf",)
    plt.close()

def velocity_magnitude_animation(frames_dir):
    u_prefix = "u"
    v_prefix = "v"
    output_file = "velocity_magnitude.mp4"
    def setup():
        u_frame_dict, u_frames = load_frame_files(frames_dir, u_prefix)
        frame_dict = {}
        for frame_num in u_frames:
            u_file = os.path.join(frames_dir, f"{u_prefix}_{frame_num:05d}.npy")
            v_file = os.path.join(frames_dir, f"{v_prefix}_{frame_num:05d}.npy")
            if os.path.exists(v_file):
                frame_dict[frame_num] = (u_file, v_file)
            else:
                print(f"Warning: Missing v velocity for frame {frame_num:05d}")
        available_frames = sorted(frame_dict.keys())

        vmax = 0.0
        available = sorted(frame_dict.keys())
        for n in available:
            u = np.load(frame_dict[n][0])
            v = np.load(frame_dict[n][1])
            mag = np.sqrt(u*u + v*v)
            vmax = max(vmax, mag.max())

        u0 = np.load(frame_dict[available_frames[0]][0])
        v0 = np.load(frame_dict[available_frames[0]][1])
        vel_mag0 = np.sqrt(u0**2 + v0**2)
        ny, nx = u0.shape
        fig, ax = plt.subplots(figsize=(8, 8 * ny / nx))
        im = ax.imshow(vel_mag0, cmap="viridis", vmin=0, vmax=vmax, origin="lower", animated=True)
        ax.set_aspect('equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Velocity Magnitude")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Velocity Magnitude - Frame {available_frames[0]}")
        return (frame_dict, fig, ax, im, available_frames)
    def update(i):
        frame_dict, _, ax, im, available_frames = setup.data
        frame_num = available_frames[i]
        u = np.load(frame_dict[frame_num][0])
        v = np.load(frame_dict[frame_num][1])
        vel_mag = np.sqrt(u**2 + v**2)
        im.set_array(vel_mag)
        ax.set_title(f"Velocity Magnitude - Frame {frame_num}")
        return im,
    setup.data = setup()
    return create_animation(frames_dir, None, lambda: setup.data, update, output_file=output_file)


def plot_simulation_data(directory, target=None):

    params = load_params(directory)
    dt = params["dt"]
    dump_period = params["dump"]
    def load_bin(name):
        path = os.path.join(directory, name)
        return np.fromfile(path, dtype=np.float64)

    # Load all scalar data
    fx = load_bin("forces_x.bin")[1:]
    fy = load_bin("forces_y.bin")[1:]
    ufish = load_bin("ufish.bin")[1:]
    vfish = load_bin("vfish.bin")[1:]
    re_vel = load_bin("Re_mesh_velocity.bin")[1:]
    re_vort = load_bin("Re_mesh_vorticity.bin")[1:]
    Tvals = load_bin("period.bin")[1:]

    # Compute time array
    n_frames = len(fx)  # assumes all same length
    time = np.arange(n_frames) * dt * dump_period + dt * dump_period

      # Compute dimensionless quantities
    denom_force = 0.5 * rho * (Lfish ** 3) / (T ** 2)
    CT = fx / denom_force
    CL = fy / denom_force

    U_star = ufish * T / Lfish
    V_star = vfish * T / Lfish
    targetstar = target * T / Lfish

    plt.figure(figsize=(10, 4))
    plt.plot(time, Tvals, label="T")
    plt.xlabel(r"$t^*$ [-]")
    plt.ylabel("Period [-]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figures/period.pdf")

    # Plot 2: Dimensionless Fish Velocities
    plt.figure(figsize=(10, 4))
    plt.plot(time, U_star, label=r"$U^*$")
    # plt.title("Dimensionless Fish Velocities")
    plt.xlabel(r"$t^*$ [-]")
    plt.ylabel("Velocity [-]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    print(targetstar, Lfish)
    plt.axhline(targetstar, color="r", linestyle='--', 
                label=f'$Target = {targetstar:.3f}$')
    plt.savefig("Figures/U_V_periodic_re1000.pdf")

        # Plot 1: Dimensionless Forces
    plt.figure(figsize=(10, 4))
    plt.plot(time, CT, label=r"$C_T$")
    plt.plot(time, CL, label=r"$C_L$")
    # plt.title("Dimensionless Hydrodynamic Forces")
    plt.xlabel(r"$t^*$ [-]")
    plt.ylabel("Forces [-]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figures/CT_CL_periodic_re1000.pdf")

    # Plot 3: Reynolds Numbers
    plt.figure(figsize=(10, 4))
    plt.plot(time, re_vel, label=r'$Re_h$')
    plt.plot(time, re_vort, label=r'$Re_{\omega}$')
    # plt.title("Mesh Reynolds Numbers")
    plt.xlabel(r"$t^*$ [-]")
    plt.ylabel("Re mesh [-]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figures/Re_mesh_periodic_re1000.pdf")

    plt.show()


def compare_T_and_Ustar(dir1, dir2, Lfish, FiguresDir="Figures"):

    for directory in (dir1, dir2):
        if not os.path.isdir(directory):
            raise ValueError(f"{directory} is not a valid directory")

    def load_bin(d, name):
        path = os.path.join(d, name)
        return np.fromfile(path, dtype=np.float64)

    data = {}
    for label, directory in (("Memory : 600", dir1), ("Memory : 200", dir2)):
        p = load_params(directory)
        dt = p["dt"]
        dump = p["dump"]
        T = 1.0
        Lfish = p["Lfish"]

        # load scalar time-series (skip first dummy entry)
        fx     = load_bin(directory, "forces_x.bin")[1:]
        ufish  = load_bin(directory, "ufish.bin")[1:]
        Tvals  = load_bin(directory, "period.bin")[1:]

        # time axis
        n = len(Tvals)
        time = np.arange(n) * dt * dump + dt * dump

        # dimensionless speed
        U_star = ufish * T / Lfish

        data[label] = dict(time=time, Tvals=Tvals, U_star=U_star)

    #--- Plot 1: Period T comparison
    plt.figure(figsize=(8,4))
    for label, d in data.items():
        plt.plot(d["time"], d["Tvals"], label=f"{label}: $T(t)$")
    plt.xlabel(r"$t^*$")
    plt.ylabel("Period $T$ [-]")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{FiguresDir}/compare_periods.pdf")
    plt.close()

    #--- Plot 2: Dimensionless speed U*
    plt.figure(figsize=(8,4))
    for label, d in data.items():
        plt.plot(d["time"], d["U_star"], label=f"{label}: $U^*(t)$")
    targetstar = -0.1 * 1.0 / Lfish
    plt.axhline(targetstar, color="r", linestyle='--', 
            label=f'$Target = {targetstar:.3f}$')
    plt.xlabel(r"$t^*$")
    plt.ylabel(r"$U^*$")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{FiguresDir}/compare_Ustar.pdf")
    plt.close()

if __name__ == "__main__":

    T = 1.0
    Lfish = 0.8
    rho = 1.0

    h = 5/4 * 1/512 
    dt = 1e-3
    dump_period = 50

    #for time in [5., 5.25, 5.5, 5.75, 6., 10.]:
    #    frame = int(time /(dt * dump_period))
    #    wuv_plot_tiled_periodic(frame_index=frame, frames_dir="dump_5000", tag="period5000")

    #mask()
    # wuv()
wuv_periodic()
    # velocity_magnitude_animation("dump_ctrl1")
    # plot_simulation_data("dump_ctrl1", target=-0.10)

    # wuv_plot_single_frame(1e-3,h,130)
