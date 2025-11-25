import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import os
import glob
import re

def steady_state(U_star, period):
    stds = np.convolve(U_star, np.ones(period)/period, mode='valid')
    return np.min(np.where(np.abs(np.gradient(stds)) < 1e-4))

def load_bin(directory, name):
    return np.fromfile(os.path.join(directory, name), dtype=np.float64)

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

def plot_cfl(direc, tag):
    params = load_params(direc)
    h = params["L"] / (params["nx"] - 3)
    dt = params["dt"]
    T = 1.0
    Lfish = params["Lfish"]
    dump = params["dump"]
    fdict_u, at_u = load_frame_files(direc, "u")
    fdict_v, at_v = load_frame_files(direc, "v")
    cfl = []
    times = np.arange(0, len(at_u)) * dt * dump

    ufish = load_bin(direc, "ufish.bin")[1:]
    U_star = ufish * T / Lfish
    period = np.where(np.isclose(times, T))[0][0] + 1
    idx_ss = steady_state(U_star, period)

    for i in range(len(at_u)):
        u = np.load(fdict_u[at_u[i]])[:-1, 1:-1]
        v = np.load(fdict_v[at_v[i]])[1:-1, :-1]
        cfl.append((np.max(np.abs(u)) + np.max(np.abs(v))) * dt / h)

    cfl = np.array(cfl)
    line, = plt.plot(times, cfl, label=f'CFL {tag}')
    avg_cfl_ss = np.mean(cfl[idx_ss:])
    print(tag, np.max(cfl))
    plt.axhline(avg_cfl_ss, color=line.get_color(), linestyle='--', 
                label=f'$CFL_{{avg}} = {avg_cfl_ss:.3f}$')

def load_cfl(direc, tag):
    params = load_params(direc)
    h = params["L"] / (params["nx"] - 3)
    dt = params["dt"]
    T = 1.0
    Lfish = params["Lfish"]
    dump = params["dump"]
    cfl = np.loadtxt(f"{direc}/cfl.txt", delimiter=',')
    times = np.arange(0, len(cfl)) * dt * dump

    ufish = load_bin(direc, "ufish.bin")[1:]
    U_star = ufish * T / Lfish
    period = np.where(np.isclose(times, T))[0][0] + 1
    idx_ss = steady_state(U_star, period)

    line, = plt.plot(times, cfl, label=f'CFL {tag}')
    avg_cfl_ss = np.mean(cfl[idx_ss:])
    print(tag, np.max(cfl))
    plt.axhline(avg_cfl_ss, color=line.get_color(), linestyle='--', 
                label=f'$CFL_{{avg}} = {avg_cfl_ss:.3f}$')

plt.figure(figsize=(12, 8))

plot_cfl("dump_period", "Periodic $Re=1000$")
load_cfl("channel_re1000", "Channel")
plot_cfl("dump_5000", "Periodic Re=5000")

plt.xlabel(r"$t^*$")
plt.ylabel("CFL")
plt.title("CFL vs Adimensional Time")
plt.grid(True)
plt.legend(ncols=3)
plt.savefig("Figures/cfl_comp.pdf")
plt.show()