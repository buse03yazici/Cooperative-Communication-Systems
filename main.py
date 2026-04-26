import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ── Parameters ──────────────────────────────────────────────────────────────
NUM_BITS = 100_000
SNR_DB   = np.arange(-5, 31, 1)
SNR_LIN  = 10 ** (SNR_DB / 10.0)
np.random.seed(42)

# ── Path Loss Model ───────────────────────────────────────────────────────────
# Relay is placed at 40% of the source-destination distance
# Path loss: channel variance = 1 / d^alpha  (alpha=3, urban environment)
ALPHA = 3
D_SD  = 1.0               # normalized total distance
D_SR  = 0.4               # source -> relay
D_RD  = 0.6               # relay  -> destination

VAR_SD = 1 / D_SD**ALPHA  # = 1.000
VAR_SR = 1 / D_SR**ALPHA  # = 15.63  (shorter link → stronger)
VAR_RD = 1 / D_RD**ALPHA  # =  4.63  (shorter link → stronger)

# ── Channel & Modulation Helpers ─────────────────────────────────────────────
def rayleigh(n, var=1.0): return np.sqrt(var/2) * (np.random.randn(n) + 1j * np.random.randn(n))
def noise(n, snr):        return np.sqrt(1/(2*snr)) * (np.random.randn(n) + 1j * np.random.randn(n))
def mod(bits):            return (2*bits - 1).astype(complex)   # BPSK: 0→-1, 1→+1
def demod(sig):           return (np.real(sig) > 0).astype(int) # hard decision
def ber(b1, b2):          return np.mean(b1 != b2)

# ── Simulation Functions ──────────────────────────────────────────────────────
def direct(snr):
    bits = np.random.randint(0, 2, NUM_BITS)
    h    = rayleigh(NUM_BITS, VAR_SD)
    rx   = h * mod(bits) + noise(NUM_BITS, snr)
    return ber(bits, demod(np.conj(h) * rx))

def af(snr):
    bits = np.random.randint(0, 2, NUM_BITS)
    tx   = mod(bits)
    h_sd = rayleigh(NUM_BITS, VAR_SD)
    h_sr = rayleigh(NUM_BITS, VAR_SR)
    h_rd = rayleigh(NUM_BITS, VAR_RD)

    y_sd = h_sd * tx + noise(NUM_BITS, snr)
    y_sr = h_sr * tx + noise(NUM_BITS, snr)

    beta     = 1 / np.sqrt(np.abs(h_sr)**2 + 1/snr)        # amplification gain
    y_rd     = h_rd * (beta * y_sr) + noise(NUM_BITS, snr)

    combined = np.conj(h_sd) * y_sd + np.conj(h_rd * h_sr) * y_rd  # MRC
    return ber(bits, demod(combined))

def df(snr):
    bits = np.random.randint(0, 2, NUM_BITS)
    tx   = mod(bits)
    h_sd = rayleigh(NUM_BITS, VAR_SD)
    h_sr = rayleigh(NUM_BITS, VAR_SR)
    h_rd = rayleigh(NUM_BITS, VAR_RD)

    y_sd     = h_sd * tx + noise(NUM_BITS, snr)
    y_sr     = h_sr * tx + noise(NUM_BITS, snr)
    relay_tx = mod(demod(np.conj(h_sr) * y_sr))             # decode & re-encode
    y_rd     = h_rd * relay_tx + noise(NUM_BITS, snr)

    combined = np.conj(h_sd) * y_sd + np.conj(h_rd) * y_rd  # MRC
    return ber(bits, demod(combined))

# ── Run ───────────────────────────────────────────────────────────────────────
print("Simulating...")
results = np.array([[direct(s), af(s), df(s)] for s in SNR_LIN])
ber_direct, ber_af, ber_df = results[:,0], results[:,1], results[:,2]
print("Done.")

# ── Plot 1: BER vs SNR ────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.semilogy(SNR_DB, 0.5*erfc(np.sqrt(SNR_LIN)), 'k--', lw=1.5, label='Theoretical BPSK (AWGN)')
plt.semilogy(SNR_DB, ber_direct, 'r-o', lw=2, ms=5, label='Direct Transmission')
plt.semilogy(SNR_DB, ber_af,     'b-s', lw=2, ms=5, label='Amplify-and-Forward (AF)')
plt.semilogy(SNR_DB, ber_df,     'g-^', lw=2, ms=5, label='Decode-and-Forward (DF)')
plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('Bit Error Rate (BER)', fontsize=13)
plt.title('BER vs SNR — Cooperative Communication Systems\n(BPSK, Rayleigh Fading + Path Loss, Single Relay)', fontsize=13)
plt.legend(fontsize=11); plt.grid(True, which='both', alpha=0.4)
plt.xlim([-5, 30]); plt.ylim([1e-5, 1])
plt.tight_layout()
plt.savefig('ber_vs_snr.png', dpi=150)

# ── Plot 2: Shannon Capacity vs SNR ──────────────────────────────────────────
cap_direct = np.log2(1 + SNR_LIN)
cap_af     = 0.5 * np.log2(1 + SNR_LIN/2 + SNR_LIN**2 / (SNR_LIN + 1))
cap_df     = 0.5 * np.log2(1 + 2*SNR_LIN)

plt.figure(figsize=(10, 6))
plt.plot(SNR_DB, cap_direct, 'r-o', lw=2, ms=5, label='Direct Transmission')
plt.plot(SNR_DB, cap_af,     'b-s', lw=2, ms=5, label='AF Cooperative')
plt.plot(SNR_DB, cap_df,     'g-^', lw=2, ms=5, label='DF Cooperative (upper bound)')
plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=13)
plt.title('Shannon Capacity vs SNR — Cooperative Communication Systems\n(Half-Duplex Relay, Single Relay Node)', fontsize=13)
plt.legend(fontsize=11); plt.grid(True, which='both', alpha=0.4)
plt.xlim([-5, 30]); plt.ylim([0, 12])
plt.tight_layout()
plt.savefig('capacity_vs_snr.png', dpi=150)

plt.show()
print("Graphs saved: ber_vs_snr.png | capacity_vs_snr.png")