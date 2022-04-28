import torch
import numpy as np
import pandas as pd
import librosa
import torchaudio as audio

class SEP28KDataset(torch.utils.data.Dataset):
    """SEP-28k Dataset."""

    def __init__(self, x, y, unsqueeze=False, transform=None):
        """
        Args:
            x (hdf5): hdf5 data one of 'Xtrain', 'Xtest', or 'Xvalid'
            y (hdf5): hdf5 file one of 'Ytrain', 'Ytest', or 'Yvalid'
            unsqueeze (bool, Optional): Whether or not to unsqueeze the feature.
              May be required for models that require image-like inputs.
            transform (callable, Optional): Optional transform to be applied
                on a sample.
        """
        self.data = x
        self.labels = y
        self.spec = audio.transforms.MelSpectrogram(n_mels=80, sample_rate=16000,
                                               n_fft=512, f_max=8000, f_min=0,
                                               power=0.5, hop_length=152, win_length=480)
        self.db = audio.transforms.AmplitudeToDB()

        self.freq_mask = audio.transforms.FrequencyMasking(freq_mask_param=1)
        # self.time_mask = audio.transforms.TimeMasking(time_mask_param=20)

        self.rng = np.random.default_rng(42)
        # self.rng_2 = np.random.default_rng(68)
        self.unsqueeze = unsqueeze

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load sliced clip
        # _, wav = wavfile.read(clip_path)
        wav = self.data[idx]
        wav = self.pad_trunc(wav, 3000, 16000).astype('float32')

        wav = torch.tensor(wav)
        wav = self.spec(wav)
        wav = self.db(wav)

        if (self.rng.choice(2,p=[0.2,0.8])):
          wav = self.freq_mask(wav)

        # if (self.rng_2.choice(2,p=[0.2,0.8])):
        #   wav = self.time_mask(wav)

        if (self.unsqueeze):
          wav = torch.unsqueeze(wav, 0)

        # get labels
        labels = self.labels[idx].astype('float32')

        # if self.transform:
        #     sample = self.transform(sample)

        return torch.tensor(wav).clone().detach(), torch.tensor(labels).clone().detach()
        
    @staticmethod
    def pad_trunc(sig, max_ms, sr):
      sig_len = sig.shape[0]
      max_len = sr//1000 * max_ms

      if (sig_len > max_len):
        # Truncate the signal to the given length
        sig = sig[:,:max_len]

      elif (sig_len < max_len):
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = np.random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        # Pad with 0s
        pad_begin = np.zeros((pad_begin_len))
        pad_end = np.zeros((pad_end_len))

        sig = np.concatenate((pad_begin, sig, pad_end), 0)
        
      return sig
