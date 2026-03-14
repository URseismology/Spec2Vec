from scipy import signal
import numpy as np
import pywt
from sklearn.preprocessing import MinMaxScaler
import numba as nb
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter1d,convolve1d

def stft_basic_spectogram(t_seg_data, sr, nperseg, overlap, window, f_min, f_max, max_normalize=True, 
                          powerlog=True, normalize_range=(1e-8,1), vmin_percentile=5, vmax_percentile=100):
    """
    STFT Spectrogram
    """
    f, t, spectro = signal.spectrogram(t_seg_data, fs=sr, nperseg=nperseg, noverlap=int(nperseg * overlap), 
                                       window=window, mode='magnitude')

    f_min_idx = (np.abs(f_min - f)).argmin()
    spectro = spectro[f_min_idx:]
    f = f[f_min_idx:]

    if f_max:
        f_max_idx = max((np.abs(f_max - f)).argmin(), f_min_idx)
        spectro = spectro[:f_max_idx]
        f = f[:f_max_idx]

    spectro_shape = spectro.shape
    if max_normalize:
      spectro = MinMaxScaler(feature_range=normalize_range).fit_transform(spectro.reshape(-1, 1)).reshape(spectro_shape)

    if powerlog:
      spectro = 10 * np.log10(spectro)

    vmax = np.percentile(spectro, vmax_percentile)  
    vmin = np.percentile(spectro, vmin_percentile)  

    spectro[spectro > vmax] = vmax
    spectro[spectro < vmin] = vmin
    f = f[::-1]
    spectro = spectro[::-1]

    return f,t,spectro


def cwt_simple(signal:np.ndarray, sr:int=1000, dt:float=None, fscale:dict={'start':1, 'end':512, 'num':100}, wavelet:str="cmor1.5-1.0", 
                vmin_percentile:int=5, vmax_percentile:int=100, f_min:int=0, f_max:int=1024, max_normalize=True, 
                fscaletype='linear', powerlog=True,
                normalize_range=(1e-8,1), decimate_factor=None):
    """
    CWT Spectrogram
    """
    assert signal.ndim==1, 'signal must be 1 dimesnional' 
    length = int(np.ceil(len(signal)/sr))

    if dt:
       sampling_period = dt
       t = np.arange(0,len(signal)*dt,dt)
    else:
      t = np.linspace(0, length, sr*length)
      sampling_period = dt if dt else np.diff(t).mean()
    
    if fscaletype=='linear':
      widths = np.arange(fscale['start'], fscale['end'], fscale['num'])
    elif fscaletype=='log':
      widths = np.geomspace(fscale['start'], fscale['end'], num=fscale['num'])
    else:
      raise ValueError('Freq. Scale type must be either linear or log')
    
    cwtmatr_cmplx, f = pywt.cwt(signal, widths, wavelet, sampling_period=sampling_period)
    spectro = np.abs(cwtmatr_cmplx[:-1, :-1]); 
    t = t[:-1]; f = f[:-1]
    #print(f.min(), f.max())

    if decimate_factor:
      spectro = spectro[:, ::decimate_factor]
      t = np.linspace(0, length, spectro.shape[1])
      #t = t[::decimate_factor]

    spectro = spectro[::-1]; 
    f = f[::-1]
    
    f_min_idx = (np.abs(f_min - f)).argmin()
    spectro = spectro[f_min_idx:]
    f = f[f_min_idx:]

    if f_max:
      f_max_idx = max((np.abs(f_max - f)).argmin(), f_min_idx)
      spectro = spectro[:f_max_idx]
      f = f[:f_max_idx]

    spectro_shape = spectro.shape
    if max_normalize:
      spectro = MinMaxScaler(feature_range=normalize_range).fit_transform(spectro.reshape(-1, 1)).reshape(spectro_shape)

    if powerlog:
      spectro = 10 * np.log10(spectro)

    vmax = np.percentile(spectro, vmax_percentile)  
    vmin = np.percentile(spectro, vmin_percentile)    

    spectro[spectro > vmax] = vmax
    spectro[spectro < vmin] = vmin

    f = f[::-1]
    spectro = spectro[::-1]

    return f,t,spectro,cwtmatr_cmplx
    
def compute_cwt(signal, fs, n_scales=80, omega0=5.0, norm='l2'):
    """CWT with configurable L1/L2 norm."""
    n = len(signal)
    n_pad = int(2 ** np.ceil(np.log2(2 * n)))
    data_pad = np.zeros(n_pad)
    data_pad[:n] = signal
    data_fft = fft(data_pad)

    freqs = np.linspace(0.5, fs/2.0 * 0.95, n_scales)
    scales = (omega0 * fs) / (2.0 * np.pi * freqs)
    output = np.zeros((len(scales), n), dtype=complex)

    for i, scale in enumerate(scales):
        x = np.arange(n_pad) - (n_pad - 1.0) / 2.0

        # Apply strict scaling norms
        if norm == 'l1':
            scale_factor = scale
        elif norm == 'l2':
            scale_factor = np.sqrt(scale)
        else:
            raise ValueError("Norm must be 'l1' or 'l2'")

        psi = (np.pi ** -0.25) * np.exp(1j * omega0 * x / scale) * np.exp(-0.5 * (x / scale) ** 2) / scale_factor
        psi_fft = fft(np.roll(psi, -(n_pad // 2)))
        output[i, :] = ifft(data_fft * np.conj(psi_fft))[:n]

    return np.abs(output), freqs

def compute_modwt_matrix(signal, level=5, wavelet='sym4'):
    """MODWT is inherently L2 normalized by the QMF filters."""
    n = len(signal)
    mod = n % (2 ** level)
    sig_pad = np.pad(signal, (0, (2**level) - mod), mode='symmetric') if mod > 0 else signal
    swt_coeffs = pywt.swt(sig_pad, wavelet, level=level, trim_approx=True)
    return np.vstack([np.abs(c)[:n] for c in swt_coeffs])

def get_gray_order(level):
    order = [0]
    for i in range(1, level + 1):
        order = order + [2**i - 1 - x for x in order]
    return order

def upsample_filter(f, level):
    zeros = 2**(level - 1) - 1
    if zeros == 0: return np.array(f)
    up = np.zeros(len(f) + (len(f) - 1) * zeros)
    up[::zeros + 1] = f
    return up

def compute_mowpt_matrix(signal, level=5, wavelet='sym4', norm='l2'):
    """MOWPT with configurable L1/L2 norm."""
    wavelet_obj = pywt.Wavelet(wavelet)
    h = np.array(wavelet_obj.dec_lo) / np.sqrt(2)
    g = np.array(wavelet_obj.dec_hi) / np.sqrt(2)

    nodes = [signal]
    for j in range(1, level + 1):
        h_up = upsample_filter(h, j)
        g_up = upsample_filter(g, j)

        next_nodes = []
        for node in nodes:
            approx = convolve1d(node, h_up, mode='wrap', origin=0)
            detail = convolve1d(node, g_up, mode='wrap', origin=0)
            next_nodes.extend([approx, detail])
        nodes = next_nodes

    freq_order = get_gray_order(level)
    matrix = np.vstack([np.abs(nodes[i]) for i in freq_order])

    if norm == 'l1':
        matrix *= (2 ** (level / 2.0))

    return matrix

@nb.njit(fastmath=True)
def atrous_circular_convolve(x, h, step):
    """
    Numba JIT compiled periodic circular convolution using the 'à trous'
    stride algorithm. Bypasses zero-padding for massive speedup.
    """
    N = len(x)
    K = len(h)
    out = np.zeros(N)
    for i in range(N):
        val = 0.0
        for j in range(K):
            # Circular boundary condition with dynamic stride
            idx = (i - j * step) % N
            val += x[idx] * h[j]
        out[i] = val
    return out


def compute_mowpt_jit(signal, level=5, wavelet='sym4', norm='l2'):
    """MOWPT utilizing JIT-compiled à trous algorithm."""
    wavelet_obj = pywt.Wavelet(wavelet)
    h = np.array(wavelet_obj.dec_lo) / np.sqrt(2)
    g = np.array(wavelet_obj.dec_hi) / np.sqrt(2)

    nodes = [signal]
    for j in range(1, level + 1):
        step = 2 ** (j - 1)
        next_nodes = []
        for node in nodes:
            approx = atrous_circular_convolve(node, h, step)
            detail = atrous_circular_convolve(node, g, step)
            next_nodes.extend([approx, detail])
        nodes = next_nodes

    freq_order = get_gray_order(level)
    matrix = np.vstack([np.abs(nodes[i]) for i in freq_order])

    if norm == 'l1':
        matrix *= (2 ** (level / 2.0))
    
    # Return raw linear coefficients
    return matrix