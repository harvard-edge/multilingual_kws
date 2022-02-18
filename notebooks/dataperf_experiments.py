# %%
import tensorflow as tf
import numpy as np
import IPython
from pathlib import Path
import matplotlib.pyplot as plt
import os
import subprocess
from sklearn.linear_model import LogisticRegression
import yaml

# %%
# %matplotlib inline

# os.chdir("..")
from multilingual_kws.embedding import transfer_learning, input_data

# %%
msdir_opus = Path.home() / "tinyspeech_harvard/dataperf/mswc_microset"
msdir_wav = Path.home() / "tinyspeech_harvard/dataperf/mswc_microset_wav"

# %%
# convert microset to wav
for language in ["en", "es"]:
    raise ValueError("caution - long process")
    for word in os.listdir(msdir_opus / language / "clips"):
        destdir = msdir_wav / language / "clips" / word
        destdir.mkdir(parents=True, exist_ok=True)
        for o in (msdir_opus / language / "clips" / word).glob("*.opus"):
            dest_file = destdir / (o.stem + ".wav")
            cmd = ["opusdec", "--rate", "16000", o, dest_file]
            subprocess.run(cmd)

# %%
em_path = (
    Path.home()
    / "tinyspeech_harvard/multilingual_embedding_wc/models/multilingual_context_73_0.8011"
)
base_model = tf.keras.models.load_model(em_path)
embedding = tf.keras.models.Model(
    name="embedding_model",
    inputs=base_model.inputs,
    outputs=base_model.get_layer(name="dense_2").output,
)
embedding.trainable = False

# %%
keyword = "bird"
keyword_samples = list(sorted((msdir_wav / "en" / "clips" / keyword).glob("*.wav")))
print(len(keyword_samples))

# %%
sample_fpath = str(keyword_samples[0])
print("Filepath:", sample_fpath)
settings = input_data.standard_microspeech_model_settings(3)
spectrogram = input_data.file2spec(settings, sample_fpath)
print("Spectrogram shape", spectrogram.shape)
# retrieve embedding vector representation (reshape into 1x49x40x1)
feature_vec = embedding.predict(spectrogram[tf.newaxis, :, :, tf.newaxis])
print("Feature vector shape:", feature_vec.shape)
plt.plot(feature_vec[0])
plt.gcf().set_size_inches(15, 5)

# %%
unknown_files_txt = Path.home() / "tinyspeech_harvard/unknown_files/unknown_files.txt"
unknown_samples_base = Path.home() / "tinyspeech_harvard/unknown_files"
unknown_files = []
unknown_en = []
with open(unknown_files_txt, "r") as fh:
    for w in fh.read().splitlines():
        unknown_files.append(unknown_samples_base / w)
        # print(w)
        lang = Path(w).parts[1]
        # print(lang)
        if lang == "en":
            unknown_en.append(unknown_samples_base / w)
print("Number of unknown files", len(unknown_files))
print("Number of unknown en", len(unknown_en))

unknown_en_words = [Path(w).parts[-2] for w in unknown_en]
print(set(unknown_en_words))
print(len(unknown_en_words) - len(set(unknown_en_words)))

# %%
# N kws, N unknown
N_SAMPLES = 5
N_TEST = 100
rng = np.random.RandomState(0)
keyword_samples = rng.choice(keyword_samples, N_SAMPLES + N_TEST, replace=False)
unknown_samples = rng.choice(unknown_files, N_SAMPLES + N_TEST, replace=False)
positive_samples = keyword_samples[:N_SAMPLES]
negative_samples = unknown_samples[:N_SAMPLES]
pos_test = keyword_samples[N_SAMPLES:]
neg_test = unknown_samples[N_SAMPLES:]

positive_spectrograms = np.array(
    [input_data.file2spec(settings, str(s)) for s in positive_samples]
)
negative_spectrograms = np.array(
    [input_data.file2spec(settings, str(s)) for s in negative_samples]
)
pos_test_spectrograms = np.array(
    [input_data.file2spec(settings, str(s)) for s in pos_test]
)
neg_test_spectrograms = np.array(
    [input_data.file2spec(settings, str(s)) for s in neg_test]
)
print(positive_spectrograms.shape, negative_spectrograms.shape)
print(pos_test_spectrograms.shape, neg_test_spectrograms.shape)

# %%
positive_fvs = embedding.predict(positive_spectrograms[:, :, :, np.newaxis])
negative_fvs = embedding.predict(negative_spectrograms[:, :, :, np.newaxis])
pos_test_fvs = embedding.predict(pos_test_spectrograms[:, :, :, np.newaxis])
neg_test_fvs = embedding.predict(neg_test_spectrograms[:, :, :, np.newaxis])

# %%
X = np.vstack([positive_fvs, negative_fvs])
print(X.shape)
y = np.hstack([np.ones(positive_fvs.shape[0]), np.zeros(negative_fvs.shape[0])])
print(y.shape)
clf = LogisticRegression(random_state=0).fit(X, y)

test_X = np.vstack([pos_test_fvs, neg_test_fvs])
test_y = np.hstack([np.ones(pos_test_fvs.shape[0]), np.zeros(neg_test_fvs.shape[0])])
print(y.shape)
print("test score", clf.score(test_X, test_y))
# 0.94

# %%
plt.hist(np.linalg.norm(pos_test_fvs, axis=1))

# %%
# performance spread

N_RUNS = 5
N_SAMPLES = 20
N_TEST = 100

settings = input_data.standard_microspeech_model_settings(3)

rng = np.random.RandomState(0)
keyword_samples = rng.choice(
    keyword_samples, (N_RUNS * N_SAMPLES) + N_TEST, replace=False
)
unknown_samples = rng.choice(unknown_files, N_SAMPLES + N_TEST, replace=False)

negative_samples = unknown_samples[:N_SAMPLES]
pos_test = keyword_samples[-N_TEST:]
neg_test = unknown_samples[-N_TEST:]

negative_spectrograms = np.array(
    [input_data.file2spec(settings, str(s)) for s in negative_samples]
)
pos_test_spectrograms = np.array(
    [input_data.file2spec(settings, str(s)) for s in pos_test]
)
neg_test_spectrograms = np.array(
    [input_data.file2spec(settings, str(s)) for s in neg_test]
)
print(pos_test_spectrograms.shape, neg_test_spectrograms.shape)

negative_fvs = embedding.predict(negative_spectrograms[:, :, :, np.newaxis])
pos_test_fvs = embedding.predict(pos_test_spectrograms[:, :, :, np.newaxis])
neg_test_fvs = embedding.predict(neg_test_spectrograms[:, :, :, np.newaxis])

test_X = np.vstack([pos_test_fvs, neg_test_fvs])
test_y = np.hstack([np.ones(pos_test_fvs.shape[0]), np.zeros(neg_test_fvs.shape[0])])

for ix in range(N_RUNS):
    print("::::: start", ix)
    start = ix * N_SAMPLES
    end = start + N_SAMPLES
    print(start, end)
    positive_samples = keyword_samples[start:end]
    positive_spectrograms = np.array(
        [input_data.file2spec(settings, str(s)) for s in positive_samples]
    )
    print(positive_spectrograms.shape, negative_spectrograms.shape)

    positive_fvs = embedding.predict(positive_spectrograms[:, :, :, np.newaxis])

    X = np.vstack([positive_fvs, negative_fvs])
    print(X.shape)
    y = np.hstack([np.ones(positive_fvs.shape[0]), np.zeros(negative_fvs.shape[0])])
    print(y.shape)
    clf = LogisticRegression(random_state=0).fit(X, y)

    print("test score", clf.score(test_X, test_y))
# %%
# %%
# YAML output
# TODO(mmaz): these are fake splits, switch to official splits

rng = np.random.RandomState(0)
shuf_pos = rng.permutation(keyword_samples)
shuf_neg = rng.permutation(unknown_en)

n_train_pos = int(len(shuf_pos) * 0.8)
n_train_neg = int(len(shuf_neg) * 0.8)
pos_train = shuf_pos[:n_train_pos]
neg_train = shuf_neg[:n_train_neg]
pos_test = shuf_pos[n_train_pos:]
neg_test = shuf_neg[n_train_neg:]

print([len(l) for l in [pos_train, neg_train, pos_test, neg_test]])

print(pos_train[0])
print(neg_train[0])
rel_pos = Path.home() / "tinyspeech_harvard/dataperf/mswc_microset_wav"
rel_neg = Path.home() / "tinyspeech_harvard/unknown_files"

# %%
settings = input_data.standard_microspeech_model_settings(3)
def get_ev(fp):
    spec = input_data.file2spec(settings, str(fp))
    return np.squeeze(embedding.predict(spec[None])).tolist()

v = get_ev(pos_train[0])
print(v)

# %%
target_train = [(str(fp.relative_to(rel_pos)), get_ev(fp)) for fp in pos_train]
nontarget_train = [(str(fp.relative_to(rel_neg)), get_ev(fp)) for fp in neg_train]
target_test = [(str(fp.relative_to(rel_pos)), get_ev(fp)) for fp in pos_test]
nontarget_test = [(str(fp.relative_to(rel_neg)), get_ev(fp)) for fp in neg_test]

# %%
def to_lod(l):
    return [dict(sample_id=fp, mswc_embedding=e) for (fp,e) in l]

train_yaml = dict(target_samples = to_lod(target_train), nontarget_samples=nontarget_train)
test_yaml = dict(target_samples = to_lod(target_test), nontarget_samples=nontarget_test)

with open("train_bird.yml", 'w') as fh:
    fh.write(yaml.dump(train_yaml, default_flow_style=None, sort_keys=False))
with open("test_bird.yml", 'w') as fh:
    fh.write(yaml.dump(test_yaml, default_flow_style=None, sort_keys=False))

# %%


