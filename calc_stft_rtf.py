import torch
import soundfile as sf
torch.set_num_threads(1)
torch.set_default_device("cpu")
import time

wavpath = "blind_data_agc/e8c87c9c-fe2e-46df-a96d-4405d59fe97d.wav"
### load wav
wavdata, sr = sf.read(wavpath)
wavdata = torch.tensor(wavdata)
### stft, istft anc calculate rtf
start = time.time()
for i in range(100):
    spec = torch.stft(wavdata, n_fft=960, hop_length=240, win_length=960, window=torch.hann_window(960), center=True, normalized=False, onesided=True, return_complex=True)
    backed = torch.istft(spec, n_fft=960, hop_length=240, win_length=960, window=torch.hann_window(960), center=True, normalized=False, onesided=True)
    end = time.time()
print(f"Time cost: {end-start}")
print("rtf: ", (end-start)/(len(wavdata) / sr * 100))