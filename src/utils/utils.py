import librosa as lr
import scipy.signal as signal
import numpy as np
import soundfile as sf

# Filter audio
def load_filter_audio(path):
    [y, fs] = lr.load(path, sr=None)
    [b,a] = signal.cheby2(N=8,
                        rs=25,
                        Wn=300,
                        btype='high',
                        fs=fs)
    y = signal.filtfilt(b,a,y)
    return [y, fs]


# RMS thresholding
def rms_threshold_audio(audio, rms_threshold, write_thresh=False, fs=30_000):
    audio_rms = lr.feature.rms(y=audio, hop_length=256, frame_length=256)[0]
    interpolated_rms = np.interp(np.arange(len(audio)), 
                                np.arange(len(audio_rms))*256, 
                                audio_rms)

    # Threshold audio
    threshold = rms_threshold
    thresholded_audio = np.where(interpolated_rms >= threshold, audio, np.nan)

    # If write_thresh is True, default fs is 30k Hz
    if write_thresh == True:
        clone = thresholded_audio.copy()
        clone[np.isnan(clone)] = 0
        print("\tWriting thresholded audio")
        sf.write("RMS_thresholded.flac", clone/32767, samplerate=fs)

    return thresholded_audio

# Get chunk indices
def get_chunks(thresholded_audio):
    start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0]
    chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))

    return start_indices, chunk_lengths

# Create thresholded hilbert envelope
def threshold_hilbert_env(section, hilbert_smooth, hilbert_threshold):
    analytic_signal = signal.hilbert(section)
    env = np.abs(analytic_signal)

    win_length = hilbert_smooth
    temp = np.ones(win_length)/win_length
    env_smooth = np.convolve(env, temp, mode='same')
    

    threshold= hilbert_threshold
    thresholded_env = np.where(env_smooth >= threshold, env_smooth, np.nan)

    return thresholded_env, env_smooth

# Merge close sections
def merge_close(env, thresholded_env, start_indices, chunk_lengths, minimum_distance):
    for start, length in zip(start_indices, chunk_lengths):
            nan_count = np.sum(np.isnan(thresholded_env[start:start+length]))
            end = start+length-nan_count
            if nan_count < minimum_distance:
                thresholded_env[start:start+length] = env[start:start+length]
            else:
                pass
    return thresholded_env

# Drop short sections within envelope
def drop_short(thresholded_env, start_indices, chunk_lengths, minimum_length):
     for start, length in zip(start_indices, chunk_lengths):
            nan_count = np.sum(np.isnan(thresholded_env[start:start+length]))
            end = start+length-nan_count

            section = thresholded_env[start:end]
            sec_length = len(section)
            
            # Using minimum length as threshold
            if sec_length < minimum_length:
                thresholded_env[start:end] = np.full(sec_length, np.nan)
     return thresholded_env
