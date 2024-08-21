import numpy as np
import pandas as pd
import scipy.signal as signal
import librosa as lr
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fftpack as fftpack
import cv2
from src.utils.spec_utils import *
import os
import toml

# Requires toml file
toml_file_path = "segment_options.toml"
with open(toml_file_path, 'r') as file:
    toml_data = toml.load(file)


def main():
    # Toml definitions
    piezo_audio_path = toml_data['paths']['piezo_audio_path']
    amb_audio_path = toml_data['paths']['amb_audio_path']
    label_path = toml_data['paths']['label_path']
    write_img = toml_data['spec']['write_img']
    write_dir = toml_data['spec']['write_dir']

    # Load Audio
    [y_piezo, fs] = lr.load(piezo_audio_path, sr=None)
    [y_amb, fs] = lr.load(amb_audio_path, sr=None)

    # Create Dataset
    df = pd.read_csv(label_path, sep='\t', header=None)
    df.rename(columns={0: 'onset', 1: 'offset', 2: 'label'}, inplace=True)
    df['onset_sample'] = (df['onset'] * fs).astype(int)
    df['offset_sample'] = (df['offset'] * fs).astype(int)
    df['length'] = df['offset_sample'] - df['onset_sample']

    category_counts = df['label'].value_counts()
    print(category_counts)

    # Iterate across rows
    corr_vals = []

    for index, row in df.iterrows():
        print(f"\t{index}")
        start = int(row['onset_sample'])
        end = int(row['offset_sample'])

        piezo_temp = y_piezo[start:end]
        amb_temp = y_amb[start:end]

        # Generate spectrograms
        stftMat = gen_spec(piezo_temp, fs)
        stftMat2 = gen_spec(amb_temp, fs)

        # Create mask and apply binary thresholding
        abs_piezo_spec = np.abs(stftMat)**2
        log_piezo = np.abs(10 * np.log10(abs_piezo_spec))
        threshold_value = 45
        _, mask = cv2.threshold(log_piezo, threshold_value, 255, cv2.THRESH_BINARY)
        mask = 255 - mask
        mask = cv2.inRange(mask, 254, 255)

        # Count number of white pixels
        num_pixels_0 = np.sum(mask == 0)
        num_pixels_255 = np.sum(mask == 255)
        #print(f"Zeros: {num_pixels_0}, 255: {num_pixels_255}")
        height, width = mask.shape
        print(f"\t{stftMat.shape}")
        
        # Apply mask
        iStftMat, S_masked = apply_mask(stftMat, mask, fs)
        iStftMat2, S_masked2 = apply_mask(stftMat2, mask, fs)
        
        duration = len(piezo_temp)/fs
        dt = 1/fs
        t = np.arange(0,duration,dt)


        # Mask
        mask = cv2.rotate(mask, cv2.ROTATE_180)
        mask = cv2.flip(mask, 1) # Flip horizontally

        # Convert spectrograms
        normalized_image3, colormap3, temp3 = spec2dB(S_masked**2) # Piezo_masked
        normalized_image4, colormap4, temp4 = spec2dB(S_masked2**2)  # Amb_masked

        # Corr coef:
        corr = np.corrcoef(normalized_image3[mask==255], normalized_image4[mask==255])
        corr_vals.append(corr[0][1])


        if write_img == True:
            normalized_image1, colormap1, temp1 = spec2dB(stftMat**2) # Piezo
            normalized_image2, colormap2, temp2 = spec2dB(stftMat2**2) # Amb
            write_gs(write_dir, str(int(row['onset_sample'])), normalized_image1, normalized_image2, normalized_image3, normalized_image4)
            temp_write = os.path.join(write_dir, str(int(row['onset_sample'])))

    # Corr Coeff
    df['corrcoeff'] = corr_vals
    df.to_csv("csv/df_ch3.csv", sep=',')

if __name__ == "__main__":
    main()