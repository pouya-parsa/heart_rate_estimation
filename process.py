import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import cv2
import numpy as np


# implementing Realt Time Heart Rate Monitoring From Facial RGB Color Video Using Webcam H.Rahman, M.U Ahmed

def extract_bpm(data_buffer_app, times, bpms):
    
    data_buffer = np.array(data_buffer_app)

    """
    Detrending
    Remove unwanted trend from series
    the collected RGB signals will be drfting and noising
    
    """
    data_buffer = signal.detrend(data_buffer, axis=0)
    
    
    # Filtering
    filter_ = np.hamming(128) * 1.4 + 0.6
#     filter_ = filter_.reshape(128, 1)
    x_filtered = filter_ * data_buffer
    
    # Normalization
#     data_buffer_normalized = (x_filtered - x_filtered.mean()) \
#                                     / x_filtered.std()
    
    data_buffer_normalized = x_filtered / np.linalg.norm(x_filtered)
    
    fft = np.fft.fft(data_buffer_normalized * 10)
    fft = np.abs(fft) ** 2
    
    times_ = np.array(times)
    
    selected_freq = (times_ > 0.75) & (times_ < 3)
    times_ = times_[selected_freq]
    
    # plt.figure()
    # plt.plot(times_, fft[selected_freq])
    # plt.title("Power of Signal by Applying FFT")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Power")
    # plt.savefig("fig.png")

    bpm = len(signal.find_peaks(fft[selected_freq][:])[0]) / (times[-1] - times[0]) * 60
    
    bpms.append(bpm)