import numpy as np
import pandas as pd
import toml
import argparse
from utils.utils import *
from utils.spec_utils import *

def main():
    # Arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('option_file', 
                        help='Path to the option file (.toml format)')
    args = parser.parse_args()

    with open(args.option_file, 'r') as file:
        toml_data = toml.load(file)

    # toml file definitions
    audio_path = toml_data['paths']['piezo_audio_path']
    rms_threshold = toml_data['parameters']['rms_threshold']
    hilbert_smooth = toml_data['parameters']['hilbert_smooth']
    hilbert_threshold = toml_data['parameters']['hilbert_threshold']
    minimum_distance = toml_data['parameters']['minimum_distance']
    minimum_length = toml_data['parameters']['minimum_length']

    audio_name = audio_path.split('.')[0]


    # Load audio using Librosa and scale using a 16 bit integer
    print(f"\tProcessing {audio_path}")
    [audio, fs] = load_filter_audio(audio_path)
    audio = audio*32767

    # Threshold using an RMS calculation
    thresholded_audio = rms_threshold_audio(audio, 
                                            rms_threshold=rms_threshold,
                                            write_thresh=False)

    # Loop through each RMS event detection to generate labels
    temp_labels = []
    [start_indices, chunk_lengths] = get_chunks(thresholded_audio)
    for start, length in zip(start_indices, chunk_lengths):
        actual_start = start
        nan_count = np.sum(np.isnan(thresholded_audio[start:start+length]))
        end = start+length-nan_count
        section = thresholded_audio[start:end]

        # Generate thresholded hilbert envelope
        [thresholded_env, env_smooth] = threshold_hilbert_env(section, hilbert_smooth, hilbert_threshold)

        # Merge close
        [start_indices_sec, chunk_lengths_sec] = get_chunks(thresholded_env)
        thresholded_env = merge_close(env_smooth, thresholded_env, start_indices_sec, chunk_lengths_sec, minimum_distance)
        
        # Drop short
        [start_indices_sec, chunk_lengths_sec] = get_chunks(thresholded_env)
        thresholded_env = drop_short(thresholded_env, start_indices_sec, chunk_lengths_sec, minimum_length)

        # Create labels
        [start_indices_sec, chunk_lengths_sec] = get_chunks(thresholded_env)
        for start, length in zip(start_indices_sec, chunk_lengths_sec):
            nan_count = np.sum(np.isnan(thresholded_env[start:start+length]))
            end = start+length-nan_count

            try:
                label = [np.around((actual_start+start)/fs,6), np.around((actual_start+end)/fs,6)]
                label = np.array([f"{x:.6f}" for x in label])
                temp_labels.append(label)
            except Exception as e:
                print(f"Error: {e}")

    # Create dataframe
    df = pd.DataFrame(temp_labels)
    df[2] = 1

    #analyze_spec()

    label_path = audio_name+'.txt'
    print(f"\tWriting: {label_path} ({len(df)} detections)")
    df.to_csv(label_path, sep='\t', index=False, header=False)

if __name__ == "__main__":
    main()