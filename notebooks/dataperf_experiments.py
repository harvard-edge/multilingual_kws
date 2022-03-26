# %%
import tensorflow as tf
import numpy as np
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import os
import subprocess
from sklearn.linear_model import LogisticRegression
import yaml
import sys
import pandas as pd
import tqdm

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
# serialized output
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
# yaml out
# def to_lod(l):
#     return [dict(sample_id=fp, mswc_embedding=e) for (fp,e) in l]

# train_yaml = dict(target_samples = to_lod(target_train), nontarget_samples=nontarget_train)
# test_yaml = dict(target_samples = to_lod(target_test), nontarget_samples=nontarget_test)

# with open("train_bird.yml", 'w') as fh:
#     yaml.dump(train_yaml, fh, default_flow_style=None, sort_keys=False)
# with open("test_bird.yml", 'w') as fh:
#     yaml.dump(test_yaml, fh, default_flow_style=None, sort_keys=False)


# %%
pb = (
    Path.home()
    / "tinyspeech_harvard/dataperf/dataperf-speech-example/selection/serialization/serialization"
)
sys.path.insert(0, str(pb))
import protoc_pb2

# %%


def serialize_vectors(fps_vecs, samples, sample_type):
    for (fp, vec) in fps_vecs:
        sample = samples.samples.add()
        sample.sample_type = sample_type
        sample.sample_id = fp
        sample.mswc_embedding_vector.extend(vec)
    return


train_samples = protoc_pb2.Samples()
serialize_vectors(target_train, train_samples, protoc_pb2.SampleType.TARGET)
serialize_vectors(nontarget_train, train_samples, protoc_pb2.SampleType.NONTARGET)

test_samples = protoc_pb2.Samples()
serialize_vectors(target_test, test_samples, protoc_pb2.SampleType.TARGET)
serialize_vectors(nontarget_test, test_samples, protoc_pb2.SampleType.NONTARGET)

# %%
with open("train.pb", "wb") as fh:
    fh.write(train_samples.SerializeToString())
with open("test.pb", "wb") as fh:
    fh.write(test_samples.SerializeToString())

# %%
target_train_np = np.array(
    [["target", fp, v] for (fp, v) in target_train], dtype=object
)
nontarget_train_np = np.array(
    [["nontarget", fp, v] for (fp, v) in nontarget_train], dtype=object
)

target_test_np = np.array([["target", fp, v] for (fp, v) in target_test], dtype=object)
nontarget_test_np = np.array(
    [["nontarget", fp, v] for (fp, v) in nontarget_test], dtype=object
)

train_np = np.vstack([target_train_np, nontarget_train_np])
test_np = np.vstack([target_test_np, nontarget_test_np])


# %%
np.savez_compressed("train_bird.npz", train=train_np)
np.savez_compressed("test_bird.npz", test=test_np)
# np.save("test.npy", test_np)
# np.save("train.npy", train_np)


# %%
load = np.load("train.npz", allow_pickle=True)["train"]
print(load.shape)
target_vecs = load[load[:, 0] == "target"][:, 2]
nontarget_vecs = load[load[:, 0] == "nontarget"][:, 2]
print(target_vecs.shape)
print(target_vecs.dtype)
print(nontarget_vecs.shape)

# %%
target_vecs[0]


# %%
samples_str = [str(f) for f in keyword_samples]
settings = input_data.standard_microspeech_model_settings(3)
spectrogram = input_data.file2spec(settings, sample_fpath)

BATCH_SIZE = 1024
AUTOTUNE = tf.data.experimental.AUTOTUNE


def to_spec(fpath):
    return input_data.file2spec(settings, fpath)


spec_ds = (
    tf.data.Dataset.from_tensor_slices(samples_str)
    .map(to_spec, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
feature_vecs = embedding.predict(spec_ds)
print("Feature vector shape:", feature_vecs.shape)
# %%

# %%

#rel_pos = Path.home() / "tinyspeech_harvard/dataperf/mswc_microset_wav"
# print(keyword_samples[0].relative_to(rel_pos))
# sample_paths = [str(fp.relative_to(rel_pos)) for fp in keyword_samples]

mswc_16khz = Path("/media/mark/hyperion/mswc/16khz_wav")
print(keyword_samples[0].relative_to(mswc_16khz))
sample_paths = [str(fp.relative_to(mswc_16khz)) for fp in keyword_samples]

print(len(sample_paths))

# %%
df = pd.DataFrame(
    data=dict(
        clip_id=sample_paths,
        #mswc_embedding_vector=pd.Series(list(feature_vec), dtype=np.float32),
        mswc_embedding_vector=pd.Series(list(feature_vecs)),
    )
)
# df = pd.DataFrame(data=dict(clip_id=sample_paths))
print(df.shape)
print(df.dtypes)
df.head()

# %%
df.to_parquet("/home/mark/tmp/try.parquet")
# %%
df = pd.read_parquet("/home/mark/tmp/try.parquet")
print(df.dtypes)
df.head()
# %%
keyword = "and"
mswc_16khz = Path("/media/mark/hyperion/mswc/16khz_wav")
keyword_samples = list(sorted((mswc_16khz / "en" / "clips" / keyword).glob("*.wav")))
print(len(keyword_samples))
print(keyword_samples[0])

# %%
mswc_embedding_destdir = Path("/media/mark/hyperion/mswc/embeddings/en")
mswc_16khz = Path("/media/mark/hyperion/mswc/16khz_wav/en/clips")
keywords = list(sorted(os.listdir(mswc_16khz)))
print(len(keywords))
for keyword in tqdm.tqdm(keywords):
    keyword_samples = list(sorted((mswc_16khz / keyword).glob("*.wav")))
    dest = mswc_embedding_destdir / f"{keyword}.parquet"
    if dest.exists():
        # we are resuming from a partial run
        continue

    samples_str = [str(f) for f in keyword_samples]
    if len(samples_str) == 0:
        # this is bad news. 
        # https://github.com/harvard-edge/multilingual_kws/issues/35
        continue

    spec_ds = (
        tf.data.Dataset.from_tensor_slices(samples_str)
        .map(to_spec, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    feature_vecs = embedding.predict(spec_ds)
    id_paths = [str(fp.relative_to(mswc_16khz)) for fp in keyword_samples]

    df = pd.DataFrame(
        data=dict(
            clip_id=id_paths,
            mswc_embedding_vector=pd.Series(list(feature_vecs)),
        )
    )
    dest = mswc_embedding_destdir / f"{keyword}.parquet"
    df.to_parquet(dest)
    # print(f"{mswc_embedding_destdir=}{keyword=}{dest=}")

# %%
keyword

# %%
uhohs = []
mswc_16khz = Path("/media/mark/hyperion/mswc/16khz_wav/en/clips")
keywords = list(sorted(os.listdir(mswc_16khz)))
print(len(keywords))
for keyword in tqdm.tqdm(keywords):
    keyword_samples = list(sorted((mswc_16khz / keyword).glob("*.wav")))
    if len(keyword_samples) == 0:
        uhohs.append(keyword)
print(len(uhohs))

# %%
print(uhohs)
# %%



# %%
mswcen = Path("/media/mark/hyperion/mswc/en/clips")
keywords = list(sorted(os.listdir(mswcen)))
rng = np.random.RandomState(0)
chosen = rng.choice(keywords, 1000, replace=False)
for keyword in chosen:
    wavs = list(sorted(Path(mswcen / keyword).iterdir()))
    if len(wavs) > 10:
        samples = rng.choice(wavs, 10, replace=False)
    else:
        samples = wavs
    for s in samples:
        shutil.copy2(s, "/media/mark/hyperion/mswc/wavtesting")
# %%

bads = []
for f in Path("/media/mark/hyperion/mswc/wavtesting").iterdir():
    if f.is_dir():
        continue
    dest = Path("/media/mark/hyperion/mswc/wavtesting") / "wavs" / (f.stem + ".wav")
    cvt = f"opusdec {f} {dest}"
    try:
        subprocess.call(cvt, shell=True, timeout=10)
    except:
        bads.append(f)

print("Bads -------------------")
for b in bads:
    print(b)
# %%
results = []
for ix, f in enumerate(Path("/media/mark/hyperion/mswc/wavtesting/wavs").iterdir()):
    cmd = f"sox {f} -n stat"
    out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # change the "\n" to " - " to inline the filename with the number of samples
    res = str(f) + "\n" + out.stdout.decode("utf8") + out.stderr.decode("utf8")
    results.append(res)
    # print("---", ix)
p = Path("/home/mark/tmp/soxout.txt")
p.write_text("\n".join(results))

# %%
rs = Path("/home/mark/tmp/soxout.txt").read_text()
qs = []
for l in rs.splitlines():
    if "Samples read" in l:
        qs.append(l)
len(qs)
# %%

# %%
good = 0
bad = 0
bads = []
for q in qs:
    if "48000" in q:
        good += 1
    else:
        bad += 1
        bads.append(q)
print(good, bad)
# %%
bads

# %%
