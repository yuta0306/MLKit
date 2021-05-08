import numpy as np
import copy
import librosa
import colorednoise as cn

class Compose:
    def __init__(self, transforms: list) -> None:
        for trns in transforms:
            assert hasattr(trns, '__call__')
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y

class OneOf:
    def __init__(self, transforms: list) -> None:
        for trns in transforms:
            assert hasattr(trns, '__call__')
        self.transforms = transforms
        
    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        return trns(y)

class GaussianNoise(object):
    def __init__(self, p=.5, max_noise_amplitude=.5, **kwargs) -> None:
        self.noise_amplitude = (0., max_noise_amplitude)
        self.p = p

    def __call__(self, y: np.ndarray):
        if np.random.random() > self.p:
            return y
        noise_amplitude = np.random.uniform(*self.noise_amplitude)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_amplitude).astype(y.dtype)
        return augmented

class GaussianNoiseSNR(object):
    def __init__(self, p=.5, min_snr=5., max_snr=20.):
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p

    def __call__(self, y: np.ndarray):
        if np.random.random() > self.p:
            return y
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented

class PinkNoiseSNR(object):
    def __init__(self, p=.5, min_snr=5., max_snr=20.):
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p

    def __call__(self, y: np.ndarray):
        if np.random.random() > self.p:
            return y
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented

class PitchShift(object):
    def __init__(self, p=.5, max_steps=5, sr=32000):
        self.max_steps = max_steps
        self.sr = sr
        self.p = p

    def __call__(self, y: np.ndarray):
        if np.random.random() > self.p:
            return y
        n_steps = np.random.randint(-self.max_steps, self.max_steps)
        augmented = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
        return augmented

class TimeStretch(object):
    def __init__(self, p=.5, max_rate=1.2):
        self.max_rate = max_rate
        self.p = p

    def __call__(self, y: np.ndarray):
        if np.random.random() > self.p:
            return y
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented

class TimeShift(object):
    def __init__(self, p=.5, max_shift_second=2, sr=32000, padding_mode="replace"):
        assert padding_mode in ["replace", "zero"], "`padding_mode` must be either 'replace' or 'zero'"
        self.max_shift_second = max_shift_second
        self.sr = sr
        self.padding_mode = padding_mode
        self.p = p

    def __call__(self, y: np.ndarray):
        if np.random.random() > self.p:
            return y
        shift = np.random.randint(-self.sr * self.max_shift_second, self.sr * self.max_shift_second)
        augmented = np.roll(y, shift)
        if self.padding_mode == "zero":
            if shift > 0:
                augmented[:shift] = 0
            else:
                augmented[shift:] = 0
        return augmented

class VolumeControl(object):
    def __init__(self, p=.5, db_limit=10, mode="uniform"):
        assert mode in ["uniform", "fade", "fade", "cosine", "sine"], \
            "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"

        self.db_limit= db_limit
        self.mode = mode
        self.p = p

    def __call__(self, y: np.ndarray):
        if np.random.random() > self.p:
            return y
        db = np.random.uniform(-self.db_limit, self.db_limit)
        if self.mode == "uniform":
            db_translated = 10 ** (db / 20)
        elif self.mode == "fade":
            lin = np.arange(len(y))[::-1] / (len(y) - 1)
            db_translated = 10 ** (db * lin / 20)
        elif self.mode == "cosine":
            cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * cosine / 20)
        else:
            sine = np.sin(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * sine / 20)
        augmented = y * db_translated
        return augmented