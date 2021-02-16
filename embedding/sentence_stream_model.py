#%%
import os
import pathlib

import tensorflow as tf
import glob
import numpy as np
import pickle

import sys

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
import input_data

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/embedding/")
import transfer_learning

#%%
data_dir = "/home/mark/tinyspeech_harvard/frequent_words/en/clips/"
model_dir = "/home/mark/tinyspeech_harvard/xfer_oov_efficientnet_binary/"
with open(model_dir + "unknown_words.pkl", "rb") as fh:
    unknown_words = pickle.load(fh)
with open(model_dir + "oov_words.pkl", "rb") as fh:
    oov_words = pickle.load(fh)
with open(model_dir + "unknown_files.pkl", "rb") as fh:
    unknown_files = pickle.load(fh)

with open(
    "/home/mark/tinyspeech_harvard/train_100_augment/" + "wordlist.txt", "r"
) as fh:
    commands = fh.read().splitlines()

print(
    len(commands), len(unknown_words), len(oov_words),
)

other_words = [
    w for w in os.listdir(data_dir) if w != "_background_noise_" and w not in commands
]
other_words.sort()
print(len(other_words))
assert len(set(other_words).intersection(commands)) == 0

#%%
#################################################
###  generate unknown files from ALL other_words
#################################################
mega_unknown_files = []
N_PER_MEGA = 134
for w in other_words:
    wavs = glob.glob(f"{data_dir}/{w}/*.wav")
    selected = np.random.choice(wavs, N_PER_MEGA, replace=False)
    mega_unknown_files.extend(selected)
np.random.shuffle(mega_unknown_files)
print(len(mega_unknown_files))

#%%

target = "merchant"
assert target not in commands, "target was used as an embedding word"
assert target not in unknown_words, "target was used as an unknown word"
assert target not in other_words, "target is present in mega_unknown_files"

sse = pathlib.Path("/home/mark/tinyspeech_harvard/streaming_sentence_experiments/")
base_dir = sse / target
model_dest_dir = base_dir / "model"
os.makedirs(model_dest_dir, exist_ok=True)
print("model dest dir", model_dest_dir)


#%%

model_settings = input_data.standard_microspeech_model_settings(3)
base_model_path = "/home/mark/tinyspeech_harvard/train_100_augment/hundredword_efficientnet_1600_selu_specaug80.0146-0.8736"
target_n_shots = os.listdir(base_dir / "n_shots")
# N_SHOTS = 5
# target_n_train = target_n_shots[:N_SHOTS]
# target_n_val = target_n_shots[N_SHOTS:]
#val_files = [str(base_dir / "n_shots" / w) for w in target_n_val]
val_names = os.listdir(base_dir / "val")

train_files = [str(base_dir / "n_shots" / w) for w in target_n_shots]
val_files = [str(base_dir / "val" / w) for w in val_names]
print("---TRAIN---", len(train_files))
print("\n".join(train_files))
print("----VAL--", len(val_files))
print("\n".join(val_files))

if val_files == []:
    val_files = train_files
    print("USING TRAIN FOR VAL!!!!")

# %%
####################
#### TRAIN MODEL
####################
name, model, details = transfer_learning.transfer_learn(
    target=target,
    train_files=train_files,
    val_files=val_files,
    #unknown_files=unknown_files,
    unknown_files=mega_unknown_files,
    num_epochs=9,
    num_batches=3,
    batch_size=64,
    model_settings=model_settings,
    base_model_path=base_model_path,
    base_model_output="dense_2",
)
print("saving", name)
model.save(model_dest_dir / name)

# %%
# sanity check model outputs
specs = [input_data.file2spec(model_settings, f) for f in val_files]
specs = np.expand_dims(specs, -1)
print(specs.shape)
preds = model.predict(specs)
amx = np.argmax(preds, axis=1)
print(amx)
print("VAL ACCURACY", amx[amx == 2].shape[0] / preds.shape[0])
print("--")

with np.printoptions(precision=3, suppress=True):
    print(preds)
