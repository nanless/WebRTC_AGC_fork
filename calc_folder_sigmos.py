import os
import scipy
import librosa

import numpy as np
import onnxruntime as ort
from enum import Enum
from tqdm import tqdm
import sys
from glob import glob
import soundfile as sf

__all__ = ["SigMOS", "Version"]


class Version(Enum):
    V1 = "v1"  # 15.10.2023


class SigMOS:
    '''
    MOS Estimator for the P.804 standard.
    See https://arxiv.org/pdf/2309.07385.pdf
    '''
    def __init__(self, model_dir, model_version=Version.V1):
        assert model_version in [v for v in Version]

        model_path_history = {
            Version.V1: os.path.join(model_dir, 'model-sigmos_1697718653_41d092e8-epo-200.onnx')
        }

        self.sampling_rate = 48_000
        self.resample_type = 'fft'
        self.model_version = model_version

        # STFT params
        self.dft_size = 960
        self.frame_size = 480
        self.window_length = 960
        self.window = np.sqrt(np.hanning(int(self.window_length) + 1)[:-1]).astype(np.float32)

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path_history[model_version], options)

    def stft(self, signal):
        last_frame = len(signal) % self.frame_size
        if last_frame == 0:
            last_frame = self.frame_size

        padded_signal = np.pad(signal, ((self.window_length - self.frame_size, self.window_length - last_frame),))
        frames = librosa.util.frame(padded_signal, frame_length=len(self.window), hop_length=self.frame_size, axis=0)
        spec = scipy.fft.rfft(frames * self.window, n=self.dft_size)
        return spec.astype(np.complex64)

    @staticmethod
    def compressed_mag_complex(x: np.ndarray, compress_factor=0.3):
        x = x.view(np.float32).reshape(x.shape + (2,)).swapaxes(-1, -2)
        x2 = np.maximum((x * x).sum(axis=-2, keepdims=True), 1e-12)
        if compress_factor == 1:
            mag = np.sqrt(x2)
        else:
            x = np.power(x2, (compress_factor - 1) / 2) * x
            mag = np.power(x2, compress_factor / 2)

        features = np.concatenate((mag, x), axis=-2)
        features = np.transpose(features, (1, 0, 2))
        return np.expand_dims(features, 0)

    def run(self, audio: np.ndarray, sr=None):
        if sr is not None and sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate, res_type=self.resample_type)
            print(f"Audio file resampled from {sr} to {self.sampling_rate}!")

        features = self.stft(audio)
        features = self.compressed_mag_complex(features)

        onnx_inputs = {inp.name: features for inp in self.session.get_inputs()}
        output = self.session.run(None, onnx_inputs)[0][0]

        result = {
            'MOS_COL': float(output[0]), 'MOS_DISC': float(output[1]), 'MOS_LOUD': float(output[2]),
            'MOS_NOISE': float(output[3]), 'MOS_REVERB': float(output[4]), 'MOS_SIG': float(output[5]),
            'MOS_OVRL': float(output[6])
        }
        return result


if __name__ == '__main__':
    ''' 
        Sample code to run the SigMOS estimator. 
        V1 (current model) is an alpha version and should be used in accordance.
    '''
    model_dir = "../SIG-Challenge/ICASSP2024/sigmos"
    data_dir = sys.argv[1]
    save_name = sys.argv[2]
    sigmos_estimator = SigMOS(model_dir=model_dir)
    files = glob(os.path.join(data_dir, "*.wav"))
    mos_dict_list = []
    for file in tqdm(files):
        wavdata, sr = sf.read(file)
        result = sigmos_estimator.run(wavdata, sr)
        result["file"] = file
        mos_dict_list.append(result)
    import pandas as pd
    df = pd.DataFrame(mos_dict_list)
    ## sum all the cols
    df["MOS_COL_MEAN"] = df["MOS_COL"].mean()
    df["MOS_DISC_MEAN"] = df["MOS_DISC"].mean()
    df["MOS_LOUD_MEAN"] = df["MOS_LOUD"].mean()
    df["MOS_NOISE_MEAN"] = df["MOS_NOISE"].mean()
    df["MOS_REVERB_MEAN"] = df["MOS_REVERB"].mean()
    df["MOS_SIG_MEAN"] = df["MOS_SIG"].mean()
    df["MOS_OVRL_MEAN"] = df["MOS_OVRL"].mean()
    df.to_csv(save_name, index=False)

    
    
