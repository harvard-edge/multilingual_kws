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

def get_attention(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    #print(list(last_hidden_states.shape))
    # should be 1,49,768 (otherwise librosa has the wrong samplerate)
    # 49: timestamps
    # 768 - attention vectors?
    return last_hidden_states

def get_embedding_from_fp(filepath):
    audio = librosa.load(filepath, sr=16000)[0]
    attention = get_attention(audio)
    return np.amax(np.squeeze(attention.numpy()), axis=0)

# %%
print(get_attention(audio).shape)
print(np.amax(np.squeeze(get_attention(audio).numpy()), 0).shape)
print(get_embedding_from_fp(wavs[0]).shape)
# %%
samples = 20

rng = np.random.RandomState(0)
fps = rng.choice(wavs, samples, replace=False)
audios = [librosa.load(fp, sr=16000)[0] for fp in fps]
embeddings = [get_attention(a) for a in audios]
print(len(audios))
print("\n".join([str(e.shape) for e in embeddings]))

# %%

print("\n".join([str(e.shape) for e in embeddings]))

# %%
x = np.concatenate(embeddings)
# x1 = np.concatenate(padded_embedding_data)
np.amax(x, 1).shape

# %%
unknown_files_txt = Path.home() / "tinyspeech_harvard/unknown_files/unknown_files.txt"
unknown_samples_base = Path.home() / "tinyspeech_harvard/unknown_files"
unknown_files = []
with open(unknown_files_txt, "r") as fh:
    for w in fh.read().splitlines():
        unknown_files.append(unknown_samples_base / w)
print("Number of unknown files", len(unknown_files))
# %%

N_RUNS = 5
N_SAMPLES = 20
N_TEST = 100

msdir_wav = Path.home() / "tinyspeech_harvard/dataperf/mswc_microset_wav"
keyword = "bird"
keyword_samples = list(sorted((msdir_wav / "en" / "clips" / keyword).glob("*.wav")))
print(len(keyword_samples))
rng = np.random.RandomState(0)
keyword_samples = rng.choice(keyword_samples, (N_RUNS * N_SAMPLES) + N_TEST, replace=False)
unknown_samples = rng.choice(unknown_files, N_SAMPLES + N_TEST, replace=False)

negative_samples = unknown_samples[:N_SAMPLES]
pos_test = keyword_samples[-N_TEST:]
neg_test = unknown_samples[-N_TEST:]

negative_fvs = np.array([get_embedding_from_fp(f) for f in negative_samples])
pos_test_fvs = np.array([get_embedding_from_fp(f) for f in pos_test])
neg_test_fvs = np.array([get_embedding_from_fp(f) for f in neg_test])

test_X = np.vstack([pos_test_fvs, neg_test_fvs])
print("testX", test_X.shape)
test_y = np.hstack([np.ones(pos_test_fvs.shape[0]), np.zeros(neg_test_fvs.shape[0])])

for ix in range(N_RUNS):
    print("::::: run", ix)
    start = ix * N_SAMPLES
    end = start + N_SAMPLES
    # print(start, end)
    positive_samples = keyword_samples[start:end]

    positive_fvs = np.array([get_embedding_from_fp(f) for f in positive_samples])

    X = np.vstack([positive_fvs, negative_fvs])
    # print(X.shape)
    y = np.hstack([np.ones(positive_fvs.shape[0]), np.zeros(negative_fvs.shape[0])])
    # print(y.shape)
    clf = sklearn.linear_model.LogisticRegression(random_state=0).fit(X, y)

    print("test score", clf.score(test_X, test_y))
# %%
#multiclass linear regression

N_RUNS = 5
N_SAMPLES = 20
N_TEST = 100

msdir_wav = Path.home() / "tinyspeech_harvard/dataperf/mswc_microset_wav"
kw1 = "bird"
kw2 = "house"
kw1s = list(sorted((msdir_wav / "en" / "clips" / kw1).glob("*.wav")))
kw2s = list(sorted((msdir_wav / "en" / "clips" / kw2).glob("*.wav")))
rng = np.random.RandomState(0)
kw1s = rng.choice(kw1s, (N_RUNS * N_SAMPLES) + N_TEST, replace=False)
kw2s = rng.choice(kw2s, (N_RUNS * N_SAMPLES) + N_TEST, replace=False)
unknown_samples = rng.choice(unknown_files, N_SAMPLES + N_TEST, replace=False)

negative_samples = unknown_samples[:N_SAMPLES]
negative_fvs = np.array([get_embedding_from_fp(f) for f in negative_samples])

pos1_test = np.array([get_embedding_from_fp(f) for f in kw1s[-N_TEST:]])
pos2_test = np.array([get_embedding_from_fp(f) for f in kw2s[-N_TEST:]])

neg_test = unknown_samples[-N_TEST:]
neg_test_fvs = np.array([get_embedding_from_fp(f) for f in neg_test])

test_X = np.vstack([pos1_test, pos2_test, neg_test_fvs])
print("testX", test_X.shape)
test_y = np.hstack([[1] * pos1_test.shape[0], [2] * pos2_test.shape[0], np.zeros(neg_test_fvs.shape[0])])

for ix in range(N_RUNS):
    print("::::: run", ix)
    start = ix * N_SAMPLES
    end = start + N_SAMPLES
    # print(start, end)
    pos1_samples = kw1s[start:end]
    pos2_samples = kw2s[start:end]
    samples = np.concatenate([pos1_samples, pos2_samples])

    positive_fvs = np.array([get_embedding_from_fp(f) for f in samples])

    X = np.vstack([positive_fvs, negative_fvs])
    # print(X.shape)
    y = np.hstack([[1] * pos1_samples.shape[0], [2] * pos2_samples.shape[0], np.zeros(negative_fvs.shape[0])])
    # print(y.shape)
    clf = sklearn.linear_model.LogisticRegression(random_state=0).fit(X, y)

    print("test score", clf.score(test_X, test_y))

# %%

# %%
