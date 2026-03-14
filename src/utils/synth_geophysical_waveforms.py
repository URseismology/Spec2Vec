import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, chirp
from dataclasses import dataclass
from typing import Tuple, Dict
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster
from scipy.spatial.distance import pdist,squareform,cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numba as nb
from scipy import signal
import os
import tempfile
import random
from collections import Counter
from scipy.stats import norm
from math import floor, log as mlog
from warnings import warn
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
from scipy.signal import chirp, gausspulse, spectrogram
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from SPEC2VEC.src.utils.gisqa_compute_updated import *

@dataclass
class SignalConfig:
    fs: float = 100.0
    duration: float = 60.0
    seed: int = 42
    noise_level: float = 0.05


class ComplexSynthetics:
    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        self.t = np.arange(0, cfg.duration, 1/cfg.fs)

    def earthquake_complex(self, p_arrival=15.0, s_arrival=35.0, p_freq=12.0, s_freq=6.0, coda_duration=40.0) -> Tuple[np.ndarray, Dict]:
        t = self.t
        signal = np.zeros_like(t)

        # P-wave
        p_end = p_arrival + 4
        p_mask = (t >= p_arrival) & (t < p_end)
        if np.any(p_mask):
            p_t = t[p_mask] - p_arrival
            flip_idx = len(p_t) // 2
            p_wave = np.sin(2 * np.pi * p_freq * p_t) * np.exp(-p_t * 1.5)
            p_wave[flip_idx:] *= -0.7
            signal[p_mask] += 0.35 * p_wave

        # S-wave
        s_end = s_arrival + 15
        s_mask = (t >= s_arrival) & (t < s_end)
        if np.any(s_mask):
            s_t = t[s_mask] - s_arrival
            s_fast = np.sin(2 * np.pi * s_freq * 1.05 * s_t) * np.exp(-s_t * 0.4)
            s_slow = np.sin(2 * np.pi * s_freq * 0.95 * s_t) * np.exp(-s_t * 0.35)
            signal[s_mask] += 0.6 * (s_fast + 0.7 * s_slow)

        # Coda
        coda_start = s_arrival + 8
        coda_mask = (t >= coda_start) & (t < coda_start + coda_duration)
        if np.any(coda_mask):
            coda_t = t[coda_mask] - coda_start
            inst_freq = s_freq * np.exp(-coda_t * 0.1)
            phase = np.cumsum(2 * np.pi * inst_freq / self.cfg.fs)
            envelope = np.exp(-coda_t * 0.15)
            coda = np.sin(phase) * envelope
            coda += 0.3 * gaussian_filter1d(np.random.randn(len(coda_t)), sigma=2) * envelope
            signal[coda_mask] += 0.25 * coda

        signal += np.random.randn(len(t)) * self.cfg.noise_level
        return signal, {'type': 'earthquake_complex'}

    def volcanic_tremor_gliding(self, base_freq=2.0, gliding_rate=0.02, harmonic_content=True) -> Tuple[np.ndarray, Dict]:
        t = self.t
        f_inst = base_freq * (1 + gliding_rate * t)
        phase = np.cumsum(2 * np.pi * f_inst / self.cfg.fs)

        tremor = np.sin(phase)
        if harmonic_content:
            tremor += 0.6 * np.sin(2 * phase)
            tremor += 0.3 * np.sin(3 * phase)

        modulation = 1 + 0.4 * np.sin(2 * np.pi * 0.15 * t)
        tremor *= modulation
        lp = 0.2 * np.sin(2 * np.pi * 0.5 * t)
        signal = tremor + lp + np.random.randn(len(t)) * 0.03
        return signal, {'type': 'tremor_gliding'}

    def volcanic_explosion_airwave(self, onset=25.0, duration=3.0, airwave_delay=10.0) -> Tuple[np.ndarray, Dict]:
        t = self.t
        signal = np.zeros_like(t)

        pre_mask = t < onset
        signal[pre_mask] += 0.1 * np.sin(2 * np.pi * 2.0 * t[pre_mask])

        body_mask = (t >= onset) & (t < onset + duration)
        if np.any(body_mask):
            body_t = t[body_mask] - onset
            body = gaussian_filter1d(np.random.randn(np.sum(body_mask)), sigma=2)
            body *= np.exp(-body_t * 1.2)
            signal[body_mask] += 0.9 * body

        air_onset = onset + airwave_delay
        air_mask = (t >= air_onset) & (t < air_onset + duration * 3)
        if np.any(air_mask):
            air_t = t[air_mask] - air_onset
            air_wave = np.sin(2 * np.pi * 1.5 * air_t) * np.exp(-air_t * 0.4)
            signal[air_mask] += 0.5 * air_wave

        signal += np.random.randn(len(t)) * 0.02
        return signal, {'type': 'explosion_airwave'}

    def hybrid_eruption_sequence(self) -> Tuple[np.ndarray, Dict]:
        t = self.t
        signal = np.zeros_like(t)

        t1_mask = t < 30
        f1 = 2.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t[t1_mask])
        phase1 = np.cumsum(2 * np.pi * f1 / self.cfg.fs)
        signal[t1_mask] += 0.3 * np.sin(phase1)

        lp_mask = (t >= 30) & (t < 35)
        if np.any(lp_mask):
            lp_t = t[lp_mask] - 30
            signal[lp_mask] += 0.8 * np.sin(2 * np.pi * 0.8 * lp_t) * np.exp(-lp_t * 0.5)

        exp_mask = (t >= 35) & (t < 38)
        if np.any(exp_mask):
            exp_t = t[exp_mask] - 35
            burst = gaussian_filter1d(np.random.randn(np.sum(exp_mask)), sigma=1)
            signal[exp_mask] += 1.0 * burst * np.exp(-exp_t * 1.0)

        coda_mask = t >= 38
        if np.any(coda_mask):
            coda_t = t[coda_mask] - 38
            coda = gaussian_filter1d(np.random.randn(np.sum(coda_mask)), sigma=5)
            signal[coda_mask] += 0.3 * coda * np.exp(-coda_t * 0.2)

        signal += np.random.randn(len(t)) * 0.02
        return signal, {'type': 'hybrid_eruption'}
    
    def earthquake_complex(self, p_arrival=15.0, s_arrival=35.0, p_freq=12.0, s_freq=6.0, coda_duration=40.0) -> Tuple[np.ndarray, Dict]:
        t = self.t
        signal = np.zeros_like(t)
        p_mask = (t >= p_arrival) & (t < p_arrival + 4)
        if np.any(p_mask):
            p_t = t[p_mask] - p_arrival
            signal[p_mask] += 0.35 * np.sin(2 * np.pi * p_freq * p_t) * np.exp(-p_t * 1.5)

        s_mask = (t >= s_arrival) & (t < s_arrival + 15)
        if np.any(s_mask):
            s_t = t[s_mask] - s_arrival
            signal[s_mask] += 0.6 * np.sin(2 * np.pi * s_freq * s_t) * np.exp(-s_t * 0.4)

        signal += np.random.randn(len(t)) * self.cfg.noise_level
        return signal, {'type': 'earthquake'}

class AdvancedSeismicSynthetics:
    def __init__(self, fs=100.0, duration=60.0, seed=101):
        self.fs = fs
        self.duration = duration
        self.t = np.arange(0, duration, 1/fs)
        np.random.seed(seed)

    def _bandpass(self, data, lowcut, highcut, order=4):
        nyq = 0.5 * self.fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return filtfilt(b, a, data)

    def t_phase_earthquake(self):
        """
        Hydroacoustic T-phase coupled to seismic.
        Characterized by emergent onset, spindle-shaped envelope, high frequency (2-15 Hz).
        """
        noise = np.random.randn(len(self.t))
        # High frequency content
        hf_signal = self._bandpass(noise, 4.0, 15.0)

        # Spindle envelope (slow buildup, slow decay)
        envelope = np.exp(-((self.t - 30) / 10)**2) * (self.t / 60)

        sig = hf_signal * envelope * 2.0
        return sig + np.random.randn(len(self.t)) * 0.05, "T-Phase EQ"

    def strombolian_explosion(self):
        """
        Impulsive, broadband acoustic/seismic shock.
        Sharp onset, rapid exponential decay.
        """
        sig = np.zeros_like(self.t)
        onset = 20.0
        mask = self.t >= onset
        if np.any(mask):
            t_sub = self.t[mask] - onset
            # Broadband burst
            burst = np.random.randn(len(t_sub))
            sig[mask] = burst * np.exp(-t_sub * 2.5) * 1.5

        return sig + np.random.randn(len(self.t)) * 0.05, "Strombolian Explosion"

    def long_period_event(self):
        """
        Fluid-driven resonance. Low frequency (0.5 - 2 Hz), slowly decaying coda.
        """
        sig = np.zeros_like(self.t)
        onset = 15.0
        mask = self.t >= onset
        if np.any(mask):
            t_sub = self.t[mask] - onset
            # Ringing fluid oscillator
            sig[mask] = np.sin(2 * np.pi * 1.2 * t_sub) * np.exp(-t_sub * 0.1) * 0.8

        return sig + np.random.randn(len(self.t)) * 0.05, "Long Period (LP)"

    def harmonic_tremor1(self):
        """
        Sustained magma movement. Narrowband fundamental with integer harmonics.
        """
        # Fundamental (2 Hz) + 2nd Harmonic (4 Hz) + 3rd (6 Hz)
        f0 = 2.0
        sig = 0.6 * np.sin(2 * np.pi * f0 * self.t)
        sig += 0.3 * np.sin(2 * np.pi * (f0 * 2) * self.t)
        sig += 0.15 * np.sin(2 * np.pi * (f0 * 3) * self.t)

        # Slow amplitude modulation
        sig *= (1 + 0.3 * np.sin(2 * np.pi * 0.05 * self.t))
        return sig + np.random.randn(len(self.t)) * 0.05, "Harmonic Tremor"

    def harmonic_tremor2(self):
        sig = 0.6 * np.sin(2 * np.pi * 2.0 * self.t)
        sig += 0.3 * np.sin(2 * np.pi * 4.0 * self.t)
        sig += 0.15 * np.sin(2 * np.pi * 6.0 * self.t)
        sig *= (1 + 0.3 * np.sin(2 * np.pi * 0.05 * self.t))
        return sig + np.random.randn(len(self.t)) * 0.05, "Harmonic Tremor"

    def hybrid_eruption(self):
        """Combines a sharp explosion (needs dt) with close harmonic tones (needs df)."""
        tremor, _ = self.harmonic_tremor2()
        explosion, _ = self.strombolian_explosion()
        # Scale them to be visible together
        return (tremor * 0.5) + explosion, "Hybrid: Tremor + Explosion"

    def hybrid_eruption2(self):
        sig = 0.4 * np.sin(2 * np.pi * 3.0 * self.t)
        exp_idx = int(25.0 * self.fs)
        exp_len = len(self.t) - exp_idx
        sig[exp_idx:] += np.random.randn(exp_len) * np.exp(-self.t[:exp_len] * 2.5) * 2.0
        return sig + np.random.randn(len(self.t)) * 0.05, "Tremor + Blast"

    def impulsive_earthquake(self):
        sig = np.zeros_like(self.t)
        p_idx, p_len = int(15.0 * self.fs), int(3.0 * self.fs)
        p_wave = np.sin(2 * np.pi * 12.0 * self.t[:p_len]) * np.exp(-self.t[:p_len] * 2.0)
        sig[p_idx:p_idx+p_len] += p_wave * 0.5

        s_idx, s_len = int(22.0 * self.fs), int(8.0 * self.fs)
        s_wave = np.sin(2 * np.pi * 4.0 * self.t[:s_len]) * np.exp(-self.t[:s_len] * 1.0)
        sig[s_idx:s_idx+s_len] += s_wave * 1.0
        return sig + np.random.randn(len(self.t)) * 0.02, "Impulsive EQ"

    def volcanic_tornillo(self):
        sig = np.zeros_like(self.t)
        onset = int(10.0 * self.fs)
        coda_len = len(self.t) - onset
        tone = np.sin(2 * np.pi * 2.5 * self.t[:coda_len])
        envelope = np.exp(-self.t[:coda_len] * 0.08)
        sig[onset:] = tone * envelope
        return sig + np.random.randn(len(self.t)) * 0.01, "Volcanic Tornillo"

    def surface_wave_dispersion(self):
        mask = (self.t >= 10.0) & (self.t <= 50.0)
        sig = np.zeros_like(self.t)
        t_sub = self.t[mask] - 10.0
        dispersed = chirp(t_sub, f0=0.5, f1=8.0, t1=40.0, method='linear')
        sig[mask] = dispersed * np.sin(np.pi * t_sub / 40.0)
        return sig + np.random.randn(len(self.t)) * 0.03, "Surface Dispersion"

class AdvancedSeismicSyntheticsParams:
    def __init__(self, fs=100.0, duration=60.0, seed=101):
        self.fs = fs
        self.duration = duration
        self.t = np.arange(0, duration, 1/fs)
        np.random.seed(seed)

    def _bandpass(self, data, lowcut, highcut, order=4):
        nyq = 0.5 * self.fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return filtfilt(b, a, data)

    def impulsive_earthquake(self, p_onset=15.0, p_dur=3.0, p_freq=12.0, p_decay=2.0, p_amp=0.5,
                            s_onset=22.0, s_dur=8.0, s_freq=4.0, s_decay=1.0, s_amp=1.0,
                            noise_level=0.02):
        sig = np.zeros_like(self.t)
        p_idx, p_len = int(p_onset * self.fs), int(p_dur * self.fs)
        p_wave = np.sin(2 * np.pi * p_freq * self.t[:p_len]) * np.exp(-self.t[:p_len] * p_decay)
        sig[p_idx:p_idx+p_len] += p_wave * p_amp

        s_idx, s_len = int(s_onset * self.fs), int(s_dur * self.fs)
        s_wave = np.sin(2 * np.pi * s_freq * self.t[:s_len]) * np.exp(-self.t[:s_len] * s_decay)
        sig[s_idx:s_idx+s_len] += s_wave * s_amp
        return sig + np.random.randn(len(self.t)) * noise_level, "Impulsive EQ"

    def volcanic_tornillo(self, onset=10.0, freq=2.5, decay=0.08, amp=1.0, noise_level=0.01):
        sig = np.zeros_like(self.t)
        onset_idx = int(onset * self.fs)
        coda_len = len(self.t) - onset_idx
        tone = np.sin(2 * np.pi * freq * self.t[:coda_len])
        envelope = np.exp(-self.t[:coda_len] * decay)
        sig[onset_idx:] = amp * tone * envelope
        return sig + np.random.randn(len(self.t)) * noise_level, "Volcanic Tornillo"

    def surface_wave_dispersion(self, start=10.0, end=50.0, f0=0.5, f1=8.0, amp=1.0, noise_level=0.03):
        mask = (self.t >= start) & (self.t <= end)
        sig = np.zeros_like(self.t)
        t_sub = self.t[mask] - start
        dispersed = chirp(t_sub, f0=f0, f1=f1, t1=(end-start), method='linear')
        sig[mask] = amp * dispersed * np.sin(np.pi * t_sub / (end-start))
        #print('noise_level:',noise_level)
        return sig + np.random.randn(len(self.t)) * noise_level, "Surface Dispersion"

    def t_phase_earthquake(self, envelope_center=30.0, envelope_width=10.0, envelope_scale=2.0,
                          hf_low=4.0, hf_high=15.0, amp=2.0, noise_level=0.05):
        noise = np.random.randn(len(self.t))
        hf_signal = self._bandpass(noise, hf_low, hf_high)
        envelope = np.exp(-((self.t - envelope_center) / envelope_width)**2) * (self.t / self.duration)
        sig = hf_signal * envelope * amp
        return sig + np.random.randn(len(self.t)) * noise_level, "T-Phase EQ"

    def strombolian_explosion(self, onset=20.0, decay=2.5, amp=1.5, noise_level=0.05):
        sig = np.zeros_like(self.t)
        mask = self.t >= onset
        if np.any(mask):
            t_sub = self.t[mask] - onset
            burst = np.random.randn(len(t_sub))
            sig[mask] = burst * np.exp(-t_sub * decay) * amp
        return sig + np.random.randn(len(self.t)) * noise_level, "Strombolian Explosion"

    def hybrid_eruption2(self, tremor_freq=3.0, tremor_amp=0.4, exp_onset=25.0, exp_decay=2.5, exp_amp=2.0, noise_level=0.05):
        sig = tremor_amp * np.sin(2 * np.pi * tremor_freq * self.t)
        exp_idx = int(exp_onset * self.fs)
        exp_len = len(self.t) - exp_idx
        if exp_len > 0:
            sig[exp_idx:] += np.random.randn(exp_len) * np.exp(-self.t[:exp_len] * exp_decay) * exp_amp
        return sig + np.random.randn(len(self.t)) * noise_level, "Tremor + Blast"

class GeophysicalSynthesizer:
    """Comprehensive hydroacoustic source library for SOFAR/CTBTO monitoring"""
    
    def __init__(self, fs=100, duration=30):
        self.fs = fs
        self.dt = 1/fs
        self.duration = duration
        self.t = np.linspace(0, duration, int(fs*duration), dtype=np.float32)
        self.n_samples = len(self.t)
        
    def tphase_eq(self):
        """Earthquake T-phase: Hyperbolic dispersion in SOFAR channel"""
        signal = np.zeros_like(self.t)
        # Multiple modal arrivals (dispersive)
        for m in range(1, 4):
            a = 5 + m * 2  # Arrival time increases with mode
            b = 100 * m    # Dispersion strength
            
            f_inst = np.zeros_like(self.t)
            valid = (self.t > a) & (self.t < a + 15)
            # Hyperbolic dispersion: high freq arrives first
            f_inst[valid] = np.sqrt(b / np.maximum(self.t[valid] - a + 0.1, 0.01))
            f_inst = np.clip(f_inst, 2, 50)
            
            phase = 2 * np.pi * np.cumsum(f_inst) / self.fs
            env = np.exp(-((self.t - (a + 3)) / 4)**2)
            signal += env * np.sin(phase) * (0.6 ** m)
        return signal.astype(np.float32)
    
    def volcanic_sofar(self):
        """Volcanic eruption in SOFAR: Long-duration harmonic tremor + explosion"""
        # Harmonic tremor (resonance in magma conduit)
        tremor = (np.sin(2 * np.pi * 4 * self.t) + 
                 0.5 * np.sin(2 * np.pi * 8 * self.t) +
                 0.25 * np.sin(2 * np.pi * 12 * self.t))
        
        # Amplitude modulation (volcanic "beats")
        modulation = 1 + 0.5 * np.sin(2 * np.pi * 0.3 * self.t)
        tremor = tremor * modulation
        
        # Explosion onset at t=10s
        explosion = np.exp(-((self.t - 10)**2) / 0.5) * np.random.randn(len(self.t))
        explosion = gaussian_filter1d(explosion, sigma=1)
        
        # Combine with different time windows
        env_tremor = np.exp(-((self.t - 15)**2) / 40)
        env_exp = np.exp(-((self.t - 10)**2) / 2)
        
        signal = tremor * env_tremor * 0.3 + explosion * env_exp * 0.8
        return signal.astype(np.float32)
    
    def whale_fin(self):
        """Fin whale: 20-Hz pulses with 1 Hz repetition"""
        pulse_duration = 1.0
        t_pulse = np.linspace(0, pulse_duration, int(self.fs * pulse_duration))
        
        # 20 Hz tone with slight downsweep
        pulse = chirp(t_pulse, f0=22, f1=18, t1=pulse_duration, method='linear')
        pulse *= np.exp(-((t_pulse - 0.5)**2) / 0.1)
        
        # Repeat every 1 second starting at t=5
        signal = np.zeros_like(self.t)
        for start in np.arange(5, 25, 1.0):
            idx = int(start * self.fs)
            if idx + len(pulse) < len(signal):
                signal[idx:idx+len(pulse)] += pulse * 0.7
        
        return signal.astype(np.float32)
    
    def whale_blue(self):
        """Blue whale B-call: Quadratic downsweep 20→12 Hz"""
        call = np.zeros_like(self.t)
        # B-call at t=10s
        start = int(10 * self.fs)
        duration_samples = int(2 * self.fs)
        
        if start + duration_samples < len(call):
            t_local = np.linspace(0, 2, duration_samples)
            freq = chirp(t_local, f0=20, f1=12, t1=2, method='quadratic')
            env = np.exp(-((t_local - 1)**2) / 0.5)
            call[start:start+duration_samples] = freq * env * 0.8
        
        return call.astype(np.float32)
    
    def icequake(self):
        """Iceberg calving: Impulsive, broadband, short duration"""
        # Multiple impulsive events
        signal = np.zeros_like(self.t)
        times = [5, 12, 20]
        
        for t0 in times:
            idx = int(t0 * self.fs)
            if idx < len(signal) - 100:
                # Ricker wavelet
                t_local = np.arange(-50, 50) / self.fs
                ricker = (1 - 2*(t_local/0.01)**2) * np.exp(-(t_local/0.01)**2)
                signal[idx-50:idx+50] += ricker * 0.9
        
        return signal.astype(np.float32)
    
    def airgun(self):
        """Seismic airgun: Impulsive, periodic, high amplitude"""
        signal = np.zeros_like(self.t)
        # Every 10 seconds
        for t0 in np.arange(2, self.duration, 10):
            idx = int(t0 * self.fs)
            if idx < len(signal):
                # Sharp impulse with bubble oscillation
                t_local = np.arange(0, int(0.5*self.fs)) / self.fs
                bubble = np.exp(-t_local/0.1) * np.sin(2 * np.pi * 10 * t_local)
                if idx + len(bubble) < len(signal):
                    signal[idx:idx+len(bubble)] += bubble * 0.8
        return signal.astype(np.float32)
    
    def shipping(self):
        """Shipping noise: Low frequency, continuous, propeller modulation"""
        # Broadband low freq
        noise = np.random.randn(len(self.t))
        noise = gaussian_filter1d(noise, sigma=5)  # <10 Hz
        
        # Propeller modulation (blade rate ~2 Hz)
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 2 * self.t)
        
        # Slow variations
        slow_env = 1 + 0.2 * np.sin(2 * np.pi * 0.1 * self.t)
        
        return (noise * modulation * slow_env * 0.4).astype(np.float32)
    
    def calibration_tone(self):
        """Calibration: Pure 37 Hz tone (common in hydrophones)"""
        tone = np.sin(2 * np.pi * 37 * self.t)
        # Intermittent
        env = np.zeros_like(self.t)
        for start in np.arange(0, self.duration, 5):
            mask = (self.t >= start) & (self.t < start + 2)
            env[mask] = 1
        return (tone * env * 0.5).astype(np.float32)
    
    def generate_all(self):
        """Return dictionary of all sources"""
        return {
            'T-Phase (Eq)': self.tphase_eq(),
            'Volcanic Tremor': self.volcanic_sofar(),
            'Fin Whale': self.whale_fin(),
            'Blue Whale': self.whale_blue(),
            'Icequake': self.icequake(),
            'Airgun': self.airgun(),
            'Shipping': self.shipping(),
            'Calib 37Hz': self.calibration_tone()
        }


 
