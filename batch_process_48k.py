import os
from scipy import interpolate
import numpy as np
import soundfile as sf
import librosa
from glob import glob

src_folder = "/data3/zhounan/codes/github_repos/SIG-Challenge/ICASSP2024/blind_data"
temp_folder = "blind_data_16k"
temp_folder2 = "blind_data_16k_agc"
dst_folder = "blind_data_agc_adaptive_gain0_target10"

os.makedirs(temp_folder, exist_ok=True)
os.makedirs(temp_folder2, exist_ok=True)
os.makedirs(dst_folder, exist_ok=True)

for wavpath in glob(os.path.join(src_folder, "*.wav")):
    wavname = os.path.basename(wavpath)
    temppath = os.path.join(temp_folder, wavname)
    temppath2 = os.path.join(temp_folder2, wavname)
    dstpath = os.path.join(dst_folder, wavname)
    wavdata, sr = sf.read(wavpath)
    wavdata_16k = librosa.resample(wavdata, orig_sr=sr, target_sr=16000)
    wavdata_16k_int16 = (wavdata_16k * 32768).astype(np.int16)
    sf.write(temppath, wavdata_16k_int16, 16000)
    os.system(f"build/agc {temppath} {temppath2}")
    wavdata_16k_agc, _ = sf.read(temppath2)
    if len(wavdata_16k_agc) > len(wavdata_16k):
        wavdata_16k_agc = wavdata_16k_agc[:len(wavdata_16k)]
    else:
        wavdata_16k_agc = np.pad(wavdata_16k_agc, (0, len(wavdata_16k) - len(wavdata_16k_agc)), mode='constant')
    gains_16k = wavdata_16k_agc / (wavdata_16k + 1e-5)
    outgains_16k = np.pad(gains_16k, (159, 0), mode='edge')
    window = np.arange(1, 161) / (np.arange(1, 161).sum())
    window = np.flip(window)
    newgains_16k = np.convolve(outgains_16k, window, mode='valid')
    ### interpolate gains from 16k to 48k
    gains = interpolate.interp1d(np.arange(len(newgains_16k)), newgains_16k, kind='linear')(np.linspace(0, len(newgains_16k)-1, len(wavdata)))
    ### plot gains_16k and gains
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.subplots(2, 1, figsize=(50, 50))
    # plt.subplot(2, 1, 1)
    # plt.plot(gains_16k[16000:16000+3200])
    # plt.subplot(2, 1, 2)
    # plt.plot(gains[48000:48000+3200*3])
    # plt.savefig("gains.png")
    # import ipdb; ipdb.set_trace()
    ### smooth the gain values, use the previous points to smooth the current point
    
    # ### use uneven window to smooth the gains
    # window = np.arange(1, 481) / (np.arange(1, 481).sum())
    # newgains = np.convolve(outgains, window, mode='valid')
    # import ipdb; ipdb.set_trace()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.subplots(3, 1, figsize=(50, 50))
    # plt.subplot(3, 1, 1)
    # plt.plot(gains_16k)
    # plt.subplot(3, 1, 2)
    # plt.plot(gains)
    # plt.subplot(3, 1, 3)
    # plt.plot(newgains)
    # plt.savefig("gains.png")
    wavdata_agc = wavdata * gains
    sf.write(dstpath, wavdata_agc, sr)
    print(wavname)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.subplots(3, 1, figsize=(50, 50))
    # plt.subplot(3, 1, 1)
    # plt.plot(gains_16k)
    # plt.subplot(3, 1, 2)
    # plt.plot(newgains_16k)
    # plt.subplot(3, 1, 3)
    # plt.plot(gains)
    # plt.savefig("gains.png")