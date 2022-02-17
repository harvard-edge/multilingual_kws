# %%
import sklearn.linear_model
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
import scipy.io.wavfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# %%
keyword = "bird"
wavdir = Path.home() / "tinyspeech_harvard/dataperf/mswc_microset_wav/en/clips" / keyword
wavs = list(sorted(wavdir.glob("*.wav")))

#%%
rng = np.random.RandomState(0)
fp = rng.choice(wavs)
print(fp)
wav = librosa.load(fp, sr=16000)
audio = wav[0]
print(audio.shape)
rate, data = scipy.io.wavfile.read(fp)
# see datatype notes for wav files
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
print(data.dtype)

# https://stackoverflow.com/a/62298670
max_int16=2**15
print(np.allclose(audio, data / max_int16))

# wav2vec2: raw_speech (np.ndarray, List[float], List[np.ndarray], List[List[float]]) — The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float values, a list of numpy arrays or a list of list of float values.
# https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__.raw_speech

# %%

# call with do_normalize=True?
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# %%

def get_embedding(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    #print(list(last_hidden_states.shape))
    # should be 1,49,768 (otherwise librosa has the wrong samplerate)
    # 49: timestamps
    # 768 - attention vectors?
    return last_hidden_states

# %%
get_embedding(audio).shape

# %%
samples = 20

rng = np.random.RandomState(0)
fps = rng.choice(wavs, samples, replace=False)
audios = [librosa.load(fp, sr=16000)[0] for fp in fps]
embeddings = [get_embedding(a) for a in audios]
print(len(audios))
print("\n".join([str(e.shape) for e in embeddings]))

# %%

max_timesteps = max([embedding.shape[1] for embedding in embeddings])
padded_embedding_data = [np.pad(embedding, ((0,0), (0, max_timesteps - embedding.shape[1]), (0, 0))) for embedding in embeddings]
# %%
print("\n".join([str(e.shape) for e in embeddings]))
print("\n".join([str(e.shape) for e in padded_embedding_data]))

# %%
x = np.concatenate(embeddings)
# x1 = np.concatenate(padded_embedding_data)
np.amax(x, 1).shape

# %%
