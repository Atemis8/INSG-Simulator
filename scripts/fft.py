import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.fft import rfft, rfftfreq

def steady_state(U_star, period):
    stds = np.convolve(U_star, np.ones(period)/period, mode='valid')
    return np.min(np.where(np.abs(np.gradient(stds)) < 1e-4))

def load_params(directory):
    param_file = os.path.join(directory, "sim.params")
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            if '=' not in line: continue
            k,v = line.strip().split('=')
            params[k] = float(v) if ('.' in v or 'e' in v.lower()) else int(v)
    return params

def load_bin(directory, name):
    return np.fromfile(os.path.join(directory, name), dtype=np.float64)

def analyze_case(directory):
    # --- load & nondimensionalize ---
    p = load_params(directory)
    Lfish, dt, dump = p["Lfish"], p["dt"], p["dump"]
    rho, T = 1.0, 1.0

    ufish = load_bin(directory, "ufish.bin")[1:]
    vfish = load_bin(directory, "vfish.bin")[1:]

    n = len(ufish)
    time = np.arange(n) * dt * dump + dt * dump
    period = np.where(np.isclose(time, T))[0][0] + 1

    U_star = ufish * T / Lfish
    V_star = vfish * T / Lfish

    # --- detect steady start ---
    idx_ss = steady_state(U_star, period)
    t_ss = idx_ss * dt * dump

    # --- FFT on steady portion of U_star ---
    sample = U_star[idx_ss:]
    N = len(sample)
    freqs = rfftfreq(N, dt * dump)
    fftv = rfft(sample)
    fft_amp = np.abs(fftv) / N
    fft_amp[1:-1] *= 2

    return time, U_star, V_star, idx_ss, t_ss, freqs, fft_amp
def plot_all(directories, labels, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    fft_data = []

    # 1) Plot each velocity time‐series separately
    for d, lab in zip(directories, labels):
        time, U_star, V_star, idx_ss, t_ss, freqs, fft_amp = analyze_case(d)

        # create a fresh figure
        fig = plt.figure(figsize=(10,4))
        plt.plot(time, U_star, label=r"$U^*$")
        plt.plot(time, V_star, label=r"$V^*$")
        plt.axvline(t_ss, color="red", linestyle="--", label=f"Steady state ($t^*\\approx{t_ss:.3f})$")
        plt.axhline(-fft_amp[0], color="green", linestyle="--", label=f"$\\overline{{U^*}}_{{steady}}\\approx{-fft_amp[0]:.3f}$")
        #plt.title(f"Velocities {lab}")
        plt.xlabel(r"$t^*$ [-]")
        plt.ylabel("Velocity [-]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # save this velocity figure
        fname = os.path.join(outdir, f"{lab.replace('=', '')}_velocities.pdf")
        fig.savefig(fname, dpi=300)
        print(f"Saved velocity plot: {fname}")
        plt.close(fig)

        fft_data.append((freqs, fft_amp, lab))

    # 2) Plot all FFTs together
    fig = plt.figure(figsize=(8,5))
    for freqs, amp, lab in fft_data:
        plt.semilogy(freqs, amp, label=lab, marker=".")
    plt.xlim(0, 7)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("Comparison of FFT Spectrum of $U^*$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # save FFT comparison
    fft_fname = os.path.join(outdir, "FFT_comparison.pdf")
    fig.savefig(fft_fname, dpi=300)
    print(f"Saved FFT comparison plot: {fft_fname}")
    plt.close(fig)

if __name__ == "__main__":
    dirs  = ["channel_re1000", "dump_period", "dump_5000"]
    labs  = ["Channel", "Re=1000", "Re=5000"]
    plot_all(dirs, labs)
