#%%
import os
from pathlib import Path

import tensorflow as tf
import glob
import numpy as np
import pickle

import sys

import input_data

import transfer_learning

#%%
traindir = Path(f"/home/mark/tinyspeech_harvard/multilang_embedding")

# SELECT MODEL
base_model_path = (
    traindir / "models" / "multilang_resume40_resume05_resume20_resume22.007-0.7981/"
)
#"multilang_resume40_resume05_resume20.022-0.7969"
#"multilang_resume40_resume05_resume20_resume22.007-0.7981/"

model_dir = Path(f"/home/mark/tinyspeech_harvard/multilang_analysis_ooe/")
unknown_collection_path = model_dir / "unknown_collection.pkl"
with open(unknown_collection_path, "rb") as fh:
    unknown_collection = pickle.load(fh)
unknown_lang_words = unknown_collection["unknown_lang_words"]
unknown_files = unknown_collection["unknown_files"]
oov_lang_words = unknown_collection["oov_lang_words"]
commands = unknown_collection["commands"]
unknown_words = set([lw[1] for lw in unknown_lang_words])

#%%
#sse = Path("/home/mark/tinyspeech_harvard/streaming_sentence_experiments/")
#sse = Path("/home/mark/tinyspeech_harvard/multilingual_streaming_sentence_experiments/")
#sse = Path("/home/mark/tinyspeech_harvard/streaming_batch_sentences/")
sse = Path("/home/mark/tinyspeech_harvard/streaming_batch_perword/")

#for ix, target in enumerate(["kurz"]):
for ix, target in enumerate(os.listdir(sse)):
    if not os.path.isdir(sse / target):
        continue
    print("::::::::::::::::::: ",ix, target)
    if target in commands:
        print( "target was used as an embedding word")
        continue
    if target in unknown_words:
        print("target was used as an unknown word")
        continue
    # assert target not in other_words, "target is present in mega_unknown_files"

    base_dir = sse / target
    assert os.path.isdir(base_dir), f"{base_dir} not present"
    assert os.path.isdir(base_dir / "n_shots"), f"shots in {base_dir} not present - generate first"
    assert os.path.isdir(base_dir / "val"), f"val in {base_dir} not present - generate first"
    model_dest_dir = base_dir / "model"
    os.makedirs(model_dest_dir, exist_ok=False)
    print("model dest dir", model_dest_dir)


    model_settings = input_data.standard_microspeech_model_settings(3)
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

    ####################
    #### TRAIN MODEL
    ####################
    name, model, details = transfer_learning.transfer_learn(
        target=target,
        train_files=train_files,
        val_files=val_files,
        unknown_files=unknown_files,
        num_epochs=4, # 9
        num_batches=1, # 3
        batch_size=64,
        model_settings=model_settings,
        base_model_path=base_model_path,
        base_model_output="dense_2",
    )
    print("saving", name)
    model.save(model_dest_dir / name)
    print(":::::::: saved to", model_dest_dir / name)

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

# %%
