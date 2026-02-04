# All the functions to make GCC-PHAT, compute speaker pairs, etc.
import numpy as np
import scipy.signal
import scipy.fft
import soundfile as sf
from pathlib import Path
import pyfftw

def gcc_phat(u1, u2, fs=None, beta=0.8):
    """
    Compute the GCC-PHAT between u1 and u2. A peak at positive index k means that u2 happens k samples later than u1

    Args:
        beta (float): tunable weighting coefficient
            1.0 = standard PHAT
            0.0 = standard GCC
            0.8 = default (some amplitude information used)
    """
    # a peak at a positive index k means that u2 happens k later than u1
    u1 = u1 - np.mean(u1)
    u2 = u2 - np.mean(u2)

    # fast FFT size (previously next power of 2, this is better)
    n = scipy.fft.next_fast_len(len(u1))


    f, R = scipy.signal.csd(
            u1, u2, 
            fs=1.0 if fs is None else fs,
            nperseg=len(u1), 
            nfft=n,
            detrend=False # We already did this
            )

    # PHAT weighting (whitening)
    # eps term avoids div by zero
    R = R / (np.abs(R)**beta + 1e-15)

    corrn = scipy.fft.irfft(R, n=n)

    return corrn, n

def compute_standard_spectrogram(data, window_width=256, incr=128):
    """Standard STFT with zero-padding. FFT size = 2*window_width."""
    starts = range(0, len(data) - window_width + 1, incr)
    n = np.arange(window_width)
    window = 0.5 * (1 - np.cos(2 * np.pi * n / (window_width - 1)))

    sg = np.zeros((len(starts), 2*window_width), dtype=complex)
    fft_buffer = np.zeros(2 * window_width)
    
    for i, start_idx in enumerate(starts):
        fft_buffer.fill(0.0)
        center_start = window_width // 2
        fft_buffer[center_start:center_start + window_width] = window * data[start_idx:start_idx + window_width]

        fft_buffer = scipy.fft.fftshift(fft_buffer)
        fft_buffer = np.roll(fft_buffer, -1)
        sg[i, :] = scipy.fft.fft(fft_buffer)

    sg = sg[:, :window_width]
    sg = np.absolute(sg) + 1e-7
    minsg = np.min(sg)
    sg = 10*(np.log10(sg)-np.log10(minsg)) 
    return np.abs(sg)


class AudioRecording:
    def __init__(self, file_path_or_data, fs=None, lazy=False, channels=None):
        """Works on either a file path, or a numpy data array"""
        self._lazy_mode = False # For lazy loading of large files
        self._sf_handle = None
        self._channel_mask = channels

        if isinstance(file_path_or_data, (str, Path)):
            self.path = Path(file_path_or_data)

            # Open handle to sound file to get properties
            temp_handle = sf.SoundFile(self.path)
            self.fs = temp_handle.samplerate
            self.total_file_channels = temp_handle.channels

            # Validate requested channels make sense (if requested)
            if self._channel_mask is not None:
                if max(self._channel_mask) >= self.total_file_channels:
                    raise ValueError(f"Requested channel {max(self._channel_mask)} "
                                     f"but file only has {self.total_file_channels}")

            if lazy:
                # Open file but don't read it yet
                self._sf_handle = temp_handle # Keeps it open
                self._lazy_mode = True
                self.frames = self._sf_handle.frames
                self.data = None
            else:
                raw_data = temp_handle.read()
                temp_handle.close()

                if self._channel_mask is not None:
                    raw_data = raw_data[:, self._channel_mask]

                # Make sure it's (n_mics, n_samples) shape, so audio[i] gets mic i
                self.data = raw_data.T
                self.frames = self.data.shape[1]

        # Raw data not a file, need to know fs
        else:
            self.data = file_path_or_data
            self.fs = fs
            self.frames = self.data.shape[1]
            if fs is None:
                raise ValueError("Must provide fs if passing data")

    def get_window(self, start_sec, duration_sec):
        """Make a new AudioRecording instance for requested window"""
        start_frame = int(start_sec * self.fs)
        read_frames = int(duration_sec * self.fs)

        if self._lazy_mode:
            # Check boundaries
            if start_frame + read_frames > self.frames:
                raise ValueError("Requested window goes beyond end of file")

            self._sf_handle.seek(start_frame)
            chunk = self._sf_handle.read(read_frames)

            # Apply channel mask
            if self._channel_mask is not None:
                chunk = chunk[:, self._channel_mask]


            if chunk.ndim > 1:
                chunk = chunk.T

            # New instance is not lazy (has the full data)
            return AudioRecording(chunk, self.fs, channels=None)
        else:
            end_frame = start_frame + read_frames
            chunk = self.data[:, start_frame:end_frame]
            return AudioRecording(chunk, self.fs)

    def close(self):
        if self._lazy_mode and self._sf_handle is not None:
            self._sf_handle.close()
            del self._sf_handle




