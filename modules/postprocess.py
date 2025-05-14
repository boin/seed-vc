import numpy as np
import pyloudnorm as pyln
import scipy.signal as signal


def loudnorm(wav_data: np.ndarray, sr: int, target_loudness: float = -23, threshold: float = 0.99) -> tuple[np.ndarray, float]:
    """
    Loudness normalization (LUFS). return limited peak audio and original loudness

    Parameters
    ----------
    wav_data : np.ndarray
        Input audio data. (float)
    sr : int
        Sampling rate. (int)
    target_loudness : float
        Target loudness. (float) default: -23
    threshold : float
        Threshold for loudness reduction. (float) default: 0.99
    
    Returns
    -------
    limited_audio : np.ndarray
        Limited peak audio. (float)
    original_loudness : float
        Original loudness. (float)
    """
    # measure the loudness first
    meter = pyln.Meter(sr)  # create BS.1770 meter
    original_loudness: float = meter.integrated_loudness(wav_data)
    # loudness normalize audio to target_loudness dB LUFS
    wav_data = pyln.normalize.loudness(wav_data, original_loudness, target_loudness)
    # Calculate the gain reduction factor
    reduction_factor = np.where(np.abs(wav_data) > threshold, threshold / np.abs(wav_data), 1.0)
    # Apply the gain reduction to the audio signal
    limited_audio = wav_data * reduction_factor
    return limited_audio, original_loudness

def eq(wav_data: np.ndarray, sr: int) -> np.ndarray:
    """
    Equalization (Frequency Shift).

    Parameters
    ----------
    wav_data : np.ndarray
        Input audio data. (float)
    sr : int
        Sampling rate.
    """
    def enhance_frequency_band(audio, sample_rate, low_freq, high_freq, gain_factor):
        # 设计带通滤波器
        nyquist = 0.5 * sample_rate  # 奈奎斯特频率
        low = low_freq / nyquist
        high = high_freq / nyquist
        # 创建带通滤波器
        b, a = signal.butter(4, [low, high], btype="bandpass")
        # 对音频信号应用带通滤波器
        filtered_audio = signal.filtfilt(b, a, audio)
        # 增加增强的频段增益
        enhanced_audio = audio + gain_factor * filtered_audio
        # 返回增强后的音频数据
        return enhanced_audio

    low_freq = 5000
    high_freq = 10000
    gain_factor = 0.2

    # 确保音频数据是float64格式
    audio = wav_data.astype(np.float64)

    # 增强频段5000Hz-10000Hz
    enhanced_audio = enhance_frequency_band(audio, sr, low_freq, high_freq, gain_factor)

    # 输出为float64格式
    enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)  # 限制音频信号在[-1, 1]之间
    return enhanced_audio
