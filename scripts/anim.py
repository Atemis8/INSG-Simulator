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

def w_nomask(frames_dir="dump"):
    w_prefix = "w"
    output_file = "vorticity.mp4"
    dupl = (2, 2)

    w_frame_dict, frames = load_frame_files(frames_dir, w_prefix)

    if not os.path.isdir(frames_dir):
        raise SystemExit(f"Error: directory '{frames_dir}' not found.")

    w0 = np.load(w_frame_dict[frames[0]])
    ny, nx = w0.shape

    fig, ax = plt.subplots(figsize=(8, 8 * ny / nx))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('off')

    im = ax.imshow(
        np.tile(w0, dupl),
        cmap="coolwarm",
        origin="lower",
        animated=True,
        vmin=-5,
        vmax=5
    )
    ax.set_aspect("equal")

    def init():
        im.set_array(np.tile(w0, dupl))
        return [im]

    def update(i):
        w_arr = np.load(w_frame_dict[frames[i]])
        im.set_array(np.tile(w_arr, dupl))
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        interval=1000/24,
        blit=True
    )

    writer = animation.FFMpegWriter(
        fps=24,
        codec="libx264",
        extra_args=["-crf", "28", "-preset", "slow"]
    )

    print(f"Saving → {output_file} …")
    ani.save(output_file, writer=writer)
    print("Done.")
    return ani



def w_periodic(frames_dir="dump"):
    w_prefix = "w"
    m_prefix = "mask"
    output_file = "vorticity.mp4"
    dupl = (2, 2)

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

    absmax = max(
        np.max(np.abs(np.load(w_frame_dict[f])))
        for f in available_frames_w
    )
    if absmax == 0:
        absmax = 1.0

    im = ax.imshow(np.tile(w_data0, dupl), cmap="coolwarm", origin="lower",
                   animated=True, vmin=-5, vmax=5)
    
    black_cmap = ListedColormap([[0, 0, 0, 0], [0, 0, 0, 1]])
    mask = ax.imshow(np.tile(m_data0, dupl), cmap=black_cmap, origin="lower", alpha=1.0, vmin=0, vmax=1, animated=True)

    ax.set_aspect('equal')
    def update(i):
        frame_num = available_frames_w[i]
        w_arr = np.load(w_frame_dict[frame_num])
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


#w_nomask()
w_periodic()

