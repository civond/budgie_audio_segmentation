import librosa as lr
import scipy.signal as signal
import numpy as np
import pandas as pd

audio_path = "data/audioCh4_short_filtered.flac"
audio_name = audio_path.split('.')[0]

print(f"Loading: {audio_path}")
[audio, fs] = lr.load(audio_path, sr=None)
audio = audio*32767

audio_rms = lr.feature.rms(y=audio, hop_length=256, frame_length=256)[0] #original = 1024

interpolated_rms = np.interp(np.arange(len(audio)), 
                             np.arange(len(audio_rms))*256, 
                             audio_rms)

duration = len(audio)/fs
dt = 1/fs
t = np.arange(0,duration,dt)

threshold = 5

temp = np.where(interpolated_rms >= threshold, interpolated_rms, np.nan)
thresholded_audio = np.where(interpolated_rms >= threshold, audio, np.nan)
"""
clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0
print("Writing")
sf.write("temp_audio/RMS_thresholded.flac", clone/32767, samplerate=fs)
"""

# Drop short sections
start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0]
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    actual_start = start
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count

    # temp audio
    section = thresholded_audio[start:end]
    sec_length = len(section)
    if sec_length < 180:
        thresholded_audio[start:end] = np.full(sec_length, np.nan)

"""clone = thresholded_audio.copy()
clone[np.isnan(clone)] = 0
print("Writing")
sf.write("temp_audio/RMS_thresholded_noshort.flac", clone/32767, samplerate=fs)"""

temp_labels = []
end
start_indices = np.where(~np.isnan(thresholded_audio) & ~np.roll(~np.isnan(thresholded_audio), 1))[0]
chunk_lengths = np.diff(np.append(start_indices, len(thresholded_audio)))
for start, length in zip(start_indices, chunk_lengths):
    actual_start = start
    nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
    end = start+length-nan_count
    section = thresholded_audio[start:end]

    duration = len(section)/fs
    dt = 1/fs
    t = np.arange(0,duration,dt)

    analytic_signal = signal.hilbert(section)
    env = np.abs(analytic_signal)

    win_length = 5
    temp = np.ones(win_length)/win_length
    env_smooth = np.convolve(env, temp, mode='same')
    

    threshold=15
    thresholded_env = np.where(env_smooth >= threshold, env_smooth, np.nan)

    # Merge close
    start_indices_sec = np.where(~np.isnan(thresholded_env) & ~np.roll(~np.isnan(thresholded_env), 1))[0];
    chunk_lengths_sec = np.diff(np.append(start_indices_sec, len(thresholded_env)))
    for start, length in zip(start_indices_sec, chunk_lengths_sec):
        nan_count = np.sum(np.isnan(thresholded_env[start:start+length]))
        end = start+length-nan_count
        if nan_count < 90:
            thresholded_env[start:start+length] = env[start:start+length]
        else:
            pass

    # Drop short
    start_indices_sec = np.where(~np.isnan(thresholded_env) & ~np.roll(~np.isnan(thresholded_env), 1))[0];
    chunk_lengths_sec = np.diff(np.append(start_indices_sec, len(thresholded_env)))
    for start, length in zip(start_indices_sec, chunk_lengths_sec):
        nan_count = np.sum(np.isnan(thresholded_env[start:start+length]))
        end = start+length-nan_count

        section = thresholded_env[start:end]
        sec_length = len(section)
        
        if sec_length < 180:
            thresholded_env[start:end] = np.full(sec_length, np.nan)
        

    extend = 30
    start_indices_sec = np.where(~np.isnan(thresholded_env) & ~np.roll(~np.isnan(thresholded_env), 1))[0];
    chunk_lengths_sec = np.diff(np.append(start_indices_sec, len(thresholded_env)))
    for start, length in zip(start_indices_sec, chunk_lengths_sec):
        nan_count = np.sum(np.isnan(thresholded_env[start:start+length]))
        end = start+length-nan_count
        temp = thresholded_env[start:end]

        try:
            label = [np.around((actual_start+start-extend)/fs,6), np.around((actual_start+end+extend)/fs,6)]
            label = np.array([f"{x:.6f}" for x in label])
            temp_labels.append(label)
        except Exception as e:
            print(f"Error: {e}")
df = pd.DataFrame(temp_labels)
df[2] = 1

label_path = audio_name+'.txt'
print(f"Writing: {label_path}")
df.to_csv(label_path, sep='\t', index=False, header=False)