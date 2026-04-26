import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
NUM_BITS     = 100_000
SNR_DB       = np.arange(-5, 31, 1)
SNR_linear   = 10 ** (SNR_DB / 10)

np.random.seed(42)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def generate_bits(n):
    return np.random.randint(0, 2, n)

def bpsk_modulate(bits):
    """0 -> -1,  1 -> +1"""
    return 2 * bits - 1

def bpsk_demodulate(signal):
    """Hard decision"""
    return (signal > 0).astype(int)

def rayleigh_channel(n):
    """Complex Rayleigh fading coefficients, normalized E[|h|^2] = 1"""
    h = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
    return h

def awgn(signal, snr_lin):
    """Add complex AWGN. Noise variance = 1 / snr_lin per complex dim."""
    noise_std = 1 / np.sqrt(2 * snr_lin)
    noise = noise_std * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

# ─────────────────────────────────────────────
# 1. DIRECT TRANSMISSION (no relay)
# ─────────────────────────────────────────────

def simulate_direct(snr_lin):
    bits = generate_bits(NUM_BITS)
    symbols = bpsk_modulate(bits).astype(complex)

    h_sd = rayleigh_channel(NUM_BITS)
    received = h_sd * symbols
    received_noisy = awgn(received, snr_lin)

    detected = bpsk_demodulate(np.real(np.conj(h_sd) * received_noisy))
    ber = np.mean(detected != bits)
    return ber

# ─────────────────────────────────────────────
# 2. AMPLIFY-AND-FORWARD (AF)
# ─────────────────────────────────────────────

def simulate_af(snr_lin):
    bits = generate_bits(NUM_BITS)
    symbols = bpsk_modulate(bits).astype(complex)

    h_sr = rayleigh_channel(NUM_BITS)
    h_rd = rayleigh_channel(NUM_BITS)
    h_sd = rayleigh_channel(NUM_BITS)

    noise_std = 1 / np.sqrt(2 * snr_lin)

    # Source -> Relay
    y_sr = h_sr * symbols + noise_std * (np.random.randn(NUM_BITS) + 1j * np.random.randn(NUM_BITS))

    # Relay amplifies with power normalization
    amp_gain = 1 / np.sqrt(np.abs(h_sr)**2 + 1 / snr_lin)
    y_relay_tx = amp_gain * y_sr

    # Relay -> Destination
    y_rd = h_rd * y_relay_tx + noise_std * (np.random.randn(NUM_BITS) + 1j * np.random.randn(NUM_BITS))

    # Direct path: Source -> Destination
    y_sd = h_sd * symbols + noise_std * (np.random.randn(NUM_BITS) + 1j * np.random.randn(NUM_BITS))

    # MRC combining
    h_eff_rd = amp_gain * h_sr * h_rd
    combined = np.conj(h_sd) * y_sd + np.conj(h_eff_rd) * y_rd

    detected = bpsk_demodulate(np.real(combined))
    ber = np.mean(detected != bits)
    return ber

# ─────────────────────────────────────────────
# 3. DECODE-AND-FORWARD (DF)
# ─────────────────────────────────────────────

def simulate_df(snr_lin):
    bits = generate_bits(NUM_BITS)
    symbols = bpsk_modulate(bits).astype(complex)

    h_sr = rayleigh_channel(NUM_BITS)
    h_rd = rayleigh_channel(NUM_BITS)
    h_sd = rayleigh_channel(NUM_BITS)

    noise_std = 1 / np.sqrt(2 * snr_lin)

    # Source -> Relay
    y_sr = h_sr * symbols + noise_std * (np.random.randn(NUM_BITS) + 1j * np.random.randn(NUM_BITS))
    bits_relay_decoded = bpsk_demodulate(np.real(np.conj(h_sr) * y_sr))

    # Relay re-encodes and transmits
    symbols_relay = bpsk_modulate(bits_relay_decoded).astype(complex)
    y_rd = h_rd * symbols_relay + noise_std * (np.random.randn(NUM_BITS) + 1j * np.random.randn(NUM_BITS))

    # Direct path
    y_sd = h_sd * symbols + noise_std * (np.random.randn(NUM_BITS) + 1j * np.random.randn(NUM_BITS))

    # MRC combining
    combined = np.conj(h_sd) * y_sd + np.conj(h_rd) * y_rd

    detected = bpsk_demodulate(np.real(combined))
    ber = np.mean(detected != bits)
    return ber

# ─────────────────────────────────────────────
# RUN SIMULATION
# ─────────────────────────────────────────────

print("Simulating... (this may take ~30 seconds)")

ber_direct = []
ber_af     = []
ber_df     = []

for i, snr in enumerate(SNR_linear):
    ber_direct.append(simulate_direct(snr))
    ber_af.append(simulate_af(snr))
    ber_df.append(simulate_df(snr))
    print(f"SNR = {SNR_DB[i]:4.0f} dB  |  Direct: {ber_direct[-1]:.4f}  |  AF: {ber_af[-1]:.4f}  |  DF: {ber_df[-1]:.4f}")

ber_direct = np.array(ber_direct)
ber_af     = np.array(ber_af)
ber_df     = np.array(ber_df)

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────

plt.figure(figsize=(10, 6))
plt.semilogy(SNR_DB, ber_direct, 'r-o',  label='Direct Transmission',   linewidth=2, markersize=5)
plt.semilogy(SNR_DB, ber_af,     'b-s',  label='AF Cooperative (MRC)',  linewidth=2, markersize=5)
plt.semilogy(SNR_DB, ber_df,     'g-^',  label='DF Cooperative (MRC)',  linewidth=2, markersize=5)

plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('Bit Error Rate (BER)', fontsize=13)
plt.title('BER vs SNR — Cooperative Communication Systems\n(BPSK, Rayleigh Fading Channel)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlim([-5, 30])
plt.ylim([1e-5, 1])

plt.tight_layout()
plt.savefig('ber_vs_snr_cooperative.png', dpi=150)
plt.show()

print("\nDone! Graph saved as 'ber_vs_snr_cooperative.png'")
# ─────────────────────────────────────────────
# SHANNON CAPACITY PLOT
# ─────────────────────────────────────────────

# Direct link capacity: C = log2(1 + SNR)
capacity_direct = np.log2(1 + SNR_linear)

# Cooperative capacity (AF & DF):
# Each uses 2 time slots (half-duplex relay), so multiply by 1/2
# Effective SNR at destination = SNR_sd + SNR_relay_path
# We approximate: SNR_eff_AF ≈ SNR/2 + SNR^2 / (SNR + 1)  (classical AF bound)
# DF upper bound: min(SNR_sr, SNR_sd + SNR_rd) — we use a clean approximation

snr_eff_af = SNR_linear / 2 + (SNR_linear**2) / (SNR_linear + 1)
snr_eff_df = np.minimum(SNR_linear, SNR_linear + SNR_linear)  # simplified: both hops strong

capacity_af = 0.5 * np.log2(1 + snr_eff_af)
capacity_df = 0.5 * np.log2(1 + 2 * SNR_linear)  # DF upper bound (perfect relay)

plt.figure(figsize=(10, 6))
plt.plot(SNR_DB, capacity_direct, 'r-o', linewidth=2, markersize=5, label='Direct Transmission')
plt.plot(SNR_DB, capacity_af,     'b-s', linewidth=2, markersize=5, label='AF Cooperative')
plt.plot(SNR_DB, capacity_df,     'g-^', linewidth=2, markersize=5, label='DF Cooperative (upper bound)')

plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=13)
plt.title('Shannon Capacity vs SNR — Cooperative Communication Systems\n(Half-Duplex Relay, Single Relay Node)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlim([-5, 30])
plt.ylim([0, 12])

plt.tight_layout()
plt.savefig('capacity_vs_snr_cooperative.png', dpi=150)
plt.show()

print("Capacity plot saved as 'capacity_vs_snr_cooperative.png'")
# Shannon Capacity
cap_direct = np.log2(1 + SNR_linear)
cap_af     = 0.5 * np.log2(1 + SNR_linear/2 + SNR_linear**2 / (SNR_linear + 1))
cap_df     = 0.5 * np.log2(1 + 2*SNR_linear)

plt.figure(figsize=(10, 6))
plt.plot(SNR_DB, cap_direct, 'r-o', lw=2, ms=5, label='Direct Transmission')
plt.plot(SNN_DB, cap_af,     'b-s', lw=2, ms=5, label='AF Cooperative')
plt.plot(SNR_DB, cap_df,     'g-^', lw=2, ms=5, label='DF Cooperative (upper bound)')
plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=13)
plt.title('Shannon Capacity vs SNR — Cooperative Communication Systems', fontsize=13)
plt.legend(fontsize=11); plt.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.savefig('capacity_vs_snr.png', dpi=150)
plt.show()