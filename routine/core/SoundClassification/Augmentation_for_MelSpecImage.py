import numpy as np
import copy
import librosa

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

class RightShift(object):
    '''Shift the image ot the right in time.'''
    
    def __init__(self, input_size, width_shift_range, p=1.):
        '''
        Initialize RightShift class
        
        Parameters
        ----------------
        input_size : int or tuple[int]
            The size of the image
        width_shift_range : int or float
            The width of shift range
        p : float
            The probability of augment        
        '''
        assert isinstance(input_size, (int, tuple))
        assert isinstance(width_shift_range, (int, float))
        assert isinstance(p, (float))
        
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            assert len(input_size) == 2
            self.input_size = input_size
            
        if isinstance(width_shift_range, int):
            assert width_shift_range > 0
            assert width_shift_range <= self.input_size[1]
            self.width_shift_range = width_shift_range
        else:
            assert width_shift_range > 0
            assert width_shift_range <= 1
            self.width_shift_range = int(width_shift_range * self.input_size[1])
            
        assert 0. < p <= 1.
        self.p = p
        
    def __call__(self, image):
        '''
        Augment with shift to right
        
        Parameters
        ----------------
        image : array_like
            Image array
            
        Returns
        -----------
        np.ndarray
        '''
        if np.random.random() > self.p:
            return image
        
        shifted_image = np.full(self.input_size, np.min(image), dtype='float32')
        random_pos = np.random.randint(1, self.width_shift_range)
        
        shifted_image[:, random_pos:] = copy.deepcopy(image[:, :-random_pos])
        
        return shifted_image
    
    def __repr__(self):
        return f'RightShift(\n\
    input_size = {self.input_size},\n\
    width_shift_range = {self.width_shift_range},\n\
    p = {self.p}\n\
)'

class GaussNoise(object):
    '''Add Gaussain Noise to the spectrogram image'''
    
    def __init__(self, input_size, mean=.0, std=None, p=1.):
        '''
        Initialize GaussNoise class
        
        Parameters
        ----------------
        input_size : int or tuple[int]
            The size of the image
        mean : int or float
            Mean of gaussian distribution
        std : int or float
            standard deviation of gaussian distribution
        p : float
            The probability of augment        
        '''
        assert isinstance(input_size, (int, tuple))
        assert isinstance(mean, (int, float))
        assert isinstance(std, (int, float)) or std is None
        assert isinstance(p, (float))
        
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            assert len(input_size) == 2
            self.input_size = input_size

        self.mean = mean

        if std is not None:
            assert std > 0.0
            self.std = std
        else:
            self.std = std

        assert p > 0.0 and p <= 1.0
        self.p = p
        
    def __call__(self, spectrogram):
        '''
        Augment
        
        Parameters
        ----------------
        spectrogram : array_like
            Spectrogram array
            
        Returns
        -----------
        np.ndarray
        '''
        if np.random.random() > self.p:
            return spectrogram

        # set some std value 
        min_pixel_value = np.min(spectrogram)
        if self.std is None:
            std_factor = 0.03     # factor number 
            std = np.abs(min_pixel_value*std_factor)

        # generate a white noise spectrogram
        gauss_mask = np.random.normal(self.mean, 
                                    std, 
                                    size=self.input_size).astype('float32')

        # add white noise to the sound spectrogram
        noisy_spectrogram = spectrogram + gauss_mask

        return noisy_spectrogram
    
    def __repr__(self):
        return f'GaussNoise(\n\
    input_size = {self.input_size},\n\
    mean = {self.mean},\n\
    std = {self.std},\n\
    p = {self.p}\n\
)'

class Reshape(object):
    """Reshape the image array."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def __call__(self, image):
        return image.reshape(self.output_size)

class MelSpec(object):
    """
    Computing MelSpectrogram
    
    Parameters
    ----------
    sr: int
        Sampling rate
    n_mels: int
        The number of melody
    fmin: int
        The minimum frequency
    fmax: int
        The maximum frequency
    """
    def __init__(self, sr: int, n_mels: int, fmin: int, fmax: int, **kwargs) -> None:
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        kwargs['n_fft'] = kwargs.get('n_fft', self.sr // 2)
        kwargs['hop_length'] = kwargs.get('hop_length', self.sr // (10 * 4))
        self.kwargs = kwargs

    def __call__(self, y) -> np.ndarray:
        melspec = librosa.melspectrogram(
            y, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
            **self.kwargs,
        )

        melspec = librosa.power_to_db(melspec).astype(np.float32)
        return melspec

        

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def crop_or_pad(y, length):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
    elif len(y) > length:
        y = y[:length]
    return y