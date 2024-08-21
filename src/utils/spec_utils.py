import numpy as np
import pandas as pd
import scipy.signal as signal
import librosa as lr
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fftpack as fftpack
import cv2
import os

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

# Create Spectrogram
def gen_spec(audio, fs):
     stftMat = lr.stft(audio, 
                    n_fft= 1024, 
                    win_length=int(fs*0.008),
                    hop_length=int(fs*0.001), 
                    center=True, 
                    window='hann')
     return stftMat

# Generate masked audio
def apply_mask(stftMat, mask, fs):
    real_part = stftMat.real * 32767
    imag_part = stftMat.imag * 32767 

    real_masked = cv2.bitwise_and(real_part, 
                                  real_part, 
                                  mask=mask)
    imag_masked = cv2.bitwise_and(imag_part, 
                                  imag_part, 
                                  mask=mask)
    # Convert the masked parts back to the original type
    real_masked = real_masked / 32767 
    imag_masked = imag_masked / 32767 

    S_masked = real_masked + 1j * imag_masked

    iStftMat = lr.istft(S_masked, 
                        n_fft=1024,
                        win_length= int(fs*0.008),
                        hop_length= int(fs*0.001), 
                        window='hann')
    return iStftMat, S_masked

# Convert spectrogramn to dB and rotate 180 degrees
def spec2dB(spec, show_img = False):
    ft_dB = lr.amplitude_to_db(np.abs(spec), ref=np.max)
    normalized_image = cv2.normalize(ft_dB, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Normalize

    temp = ft_dB.copy()
    
    normalized_image = cv2.rotate(normalized_image, cv2.ROTATE_180) # Rotate 180 degrees
    normalized_image = cv2.flip(normalized_image, 1) # Flip horizontally
    colormap = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET) # Apply colormap
    if show_img == True:
            cv2.imshow("piezo", colormap)
            
    return normalized_image, colormap, temp

def analyze_spec():
     print("help")

def write_gs(write_dir, onset, mask, normalized_image1, normalized_image2, normalized_image3, normalized_image4):
    temp_write = os.path.join(write_dir, onset)
        
    try:
        os.mkdir(temp_write)
    except Exception as e:
        #print(f"{temp_write} already exists")
        pass

    # Mask
    cv2.imwrite(os.path.join(temp_write,"mask.jpg"), mask)

    # Im 1
    cv2.imwrite(os.path.join(temp_write,"orig_piezo.jpg"), normalized_image1)
    cv2.imwrite(os.path.join(temp_write,"orig_piezo_cm.jpg"), colormap1)
    
    # Im 2
    cv2.imwrite(os.path.join(temp_write,"orig_amb.jpg"), normalized_image2)
    cv2.imwrite(os.path.join(temp_write,"orig_amb_cm.jpg"), colormap2)

    # Im 3
    cv2.imwrite(os.path.join(temp_write,"masked_piezo.jpg"), normalized_image3)
    cv2.imwrite(os.path.join(temp_write,"masked_piezo_cm.jpg"), colormap3)

    # Im 4
    cv2.imwrite(os.path.join(temp_write,"masked_amb.jpg"), normalized_image4)
    cv2.imwrite(os.path.join(temp_write,"masked_amb_cm.jpg"), colormap4)

    # Merge images for display
    #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #merged = np.concatenate((mask, colormap1, colormap2, colormap3, colormap4), axis=1)
    #cv2.imshow(f"merged_{int(row['label'])}_{int(row['onset_sample'])}", merged)
    #cv2.imwrite("temp_fig/merged_cm.jpg", merged)
    #cv2.imwrite(os.path.join(temp_write, "merged_cm.jpg"), merged)
        
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()