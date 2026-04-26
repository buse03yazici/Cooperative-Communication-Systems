import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ── Parameters & Path Loss ───────────────────────────────────────────────────
NUM_BITS = 100_000
SNR_DB   = np.arange(-5, 31, 1)
SNR_LIN  = 10 ** (SNR_DB / 10.0)
np.random.seed(42)

ALPHA, D_SD, D_SR, D_RD = 3, 1.0, 0.4, 0.6
VAR_SD, VAR_SR, VAR_RD  = 1/D_SD**ALPHA, 1/D_SR**ALPHA, 1/D_RD**ALPHA
SNR_SD_EFF, SNR_SR_EFF, SNR_RD_EFF = VAR_SD*SNR_LIN, VAR_SR*SNR_LIN, VAR_RD*SNR_LIN

# ── Helpers ──────────────────────────────────────────────────────────────────
def ray(n, v=1.):  return np.sqrt(v/2)*(np.random.randn(n)+1j*np.random.randn(n))
def wgn(n, s):     return np.sqrt(1/(2*s))*(np.random.randn(n)+1j*np.random.randn(n))
def mod(b):        return (2*b-1).astype(complex)
def dem(x):        return (np.real(x)>0).astype(int)
def ber(a,b):      return np.mean(a!=b)

# ── Simulations ──────────────────────────────────────────────────────────────
def direct(s):
    bits=np.random.randint(0,2,NUM_BITS); h=ray(NUM_BITS,VAR_SD)
    return ber(bits, dem(np.conj(h)*(h*mod(bits)+wgn(NUM_BITS,s))))

def af(s):
    bits=np.random.randint(0,2,NUM_BITS); tx=mod(bits)
    hsd,hsr,hrd = ray(NUM_BITS,VAR_SD),ray(NUM_BITS,VAR_SR),ray(NUM_BITS,VAR_RD)
    ysd=hsd*tx+wgn(NUM_BITS,s); ysr=hsr*tx+wgn(NUM_BITS,s)
    beta=1/np.sqrt(np.abs(hsr)**2+1/s)
    yrd=hrd*(beta*ysr)+wgn(NUM_BITS,s)
    nv_sd=1/s; nv_rd=(np.abs(hrd)**2*beta**2+1)/s
    comb=np.conj(hsd)/nv_sd*ysd + np.conj(hrd*beta*hsr)/nv_rd*yrd
    return ber(bits, dem(comb))

def df(s):
    bits=np.random.randint(0,2,NUM_BITS); tx=mod(bits)
    hsd,hsr,hrd = ray(NUM_BITS,VAR_SD),ray(NUM_BITS,VAR_SR),ray(NUM_BITS,VAR_RD)
    ysd=hsd*tx+wgn(NUM_BITS,s); ysr=hsr*tx+wgn(NUM_BITS,s)
    yrd=hrd*mod(dem(np.conj(hsr)*ysr))+wgn(NUM_BITS,s)
    return ber(bits, dem(np.conj(hsd)*ysd+np.conj(hrd)*yrd))

# ── Run ───────────────────────────────────────────────────────────────────────
print("Simulating...")
res = np.array([[direct(s),af(s),df(s)] for s in SNR_LIN])
bd,ba,bdf = res[:,0],res[:,1],res[:,2]
print("Done.\n")

# ── Diversity Order ───────────────────────────────────────────────────────────
def div_order(b):
    m=(SNR_DB>=15)&(b>1e-9); sl,_=np.polyfit(SNR_DB[m],np.log10(b[m]),1); return -sl*10

dd,da,ddf = div_order(bd),div_order(ba),div_order(bdf)
print(f"Diversity Orders → Direct:{dd:.2f}(≈1)  AF:{da:.2f}(≈2)  DF:{ddf:.2f}(≈2)")

# ── Plot 1: BER ───────────────────────────────────────────────────────────────
plt.figure(figsize=(10,6))
plt.semilogy(SNR_DB, .5*erfc(np.sqrt(SNR_LIN)), 'k--', lw=1.5, label='Theoretical BPSK (AWGN)')
plt.semilogy(SNR_DB, bd,  'r-o', lw=2, ms=5, label=f'Direct (d≈{dd:.1f})')
plt.semilogy(SNR_DB, ba,  'b-s', lw=2, ms=5, label=f'Amplify-and-Forward (d≈{da:.1f})')
plt.semilogy(SNR_DB, bdf, 'g-^', lw=2, ms=5, label=f'Decode-and-Forward (d≈{ddf:.1f})')
plt.xlabel('SNR (dB)',fontsize=13); plt.ylabel('BER',fontsize=13)
plt.title(f'BER vs SNR — Cooperative Comm. (BPSK, Rayleigh+Path Loss α={ALPHA})',fontsize=13)
plt.legend(fontsize=11); plt.grid(True,which='both',alpha=0.4)
plt.xlim([-5,30]); plt.ylim([1e-5,1]); plt.tight_layout()
plt.savefig('ber_vs_snr.png',dpi=150)

# ── Plot 2: Capacity ──────────────────────────────────────────────────────────
cap_d  = np.log2(1+SNR_SD_EFF)
cap_af = .5*np.log2(1+(SNR_SR_EFF*SNR_RD_EFF)/(SNR_SR_EFF+SNR_RD_EFF+1))
cap_df = .5*np.log2(1+np.minimum(SNR_SR_EFF,SNR_SD_EFF)+SNR_RD_EFF)

plt.figure(figsize=(10,6))
plt.plot(SNR_DB, cap_d,  'r-o', lw=2, ms=5, label='Direct')
plt.plot(SNR_DB, cap_af, 'b-s', lw=2, ms=5, label='AF [Laneman 2004]')
plt.plot(SNR_DB, cap_df, 'g-^', lw=2, ms=5, label='DF (upper bound)')
plt.xlabel('SNR (dB)',fontsize=13); plt.ylabel('Spectral Efficiency (bits/s/Hz)',fontsize=13)
plt.title(f'Shannon Capacity vs SNR — Half-Duplex Relay (α={ALPHA}, D_SR={D_SR}, D_RD={D_RD})',fontsize=13)
plt.legend(fontsize=11); plt.grid(True,which='both',alpha=0.4)
plt.xlim([-5,30]); plt.ylim([0,15]); plt.tight_layout()
plt.savefig('capacity_vs_snr.png',dpi=150)

plt.show()
print("Graphs saved: ber_vs_snr.png | capacity_vs_snr.png")
