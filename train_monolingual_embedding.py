import os
import pickle

import logging
import glob
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

import sys

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
import input_data

from pathlib import Path
import pickle

LANG_ISOCODE = "nl"
embedding_model_dir=f"/home/mark/tinyspeech_harvard/train_{LANG_ISOCODE}_165/"
save_models_dir=f"/home/mark/tinyspeech_harvard/train_{LANG_ISOCODE}_165/models/"
data_dir = Path(f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/clips/")
os.chdir(embedding_model_dir)

if not os.path.isdir(save_models_dir):
    raise ValueError("create model dir")
if len(os.listdir(save_models_dir)) > 0:
    raise ValueError("models already exist in ", save_models_dir)

with open("commands.txt", "r") as fh:
    commands = fh.read().splitlines()
# with open("other_words.txt", "r") as fh:
#     other_words = fh.read().splitlines()

# with open("train_val_test_data.pkl", 'rb') as fh:
#     train_val_test_data = pickle.load(fh)

with open("train_files.txt", "r") as fh:
    train_files = fh.read().splitlines()
with open("val_files.txt", "r") as fh:
    val_files = fh.read().splitlines()
with open("test_files.txt", "r") as fh:
    test_files = fh.read().splitlines()


model_settings = input_data.standard_microspeech_model_settings(label_count=166)
bg_datadir = (
    f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/_background_noise_/"
)

if not os.path.isdir(bg_datadir):
    raise ValueError("no bg data at", bg_datadir)

a = input_data.AudioDataset(
    model_settings,
    commands,
    bg_datadir,
    [],
    unknown_percentage=0,
    spec_aug_params=input_data.SpecAugParams(percentage=80),
)
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = a.init_from_parent_dir(AUTOTUNE, train_files, is_training=True)
val_ds = a.init_from_parent_dir(AUTOTUNE, val_files, is_training=False)
# test_ds = a.init_from_parent_dir(AUTOTUNE, test_files, is_training=False)
batch_size = 64
train_ds = train_ds.shuffle(buffer_size=4000).batch(batch_size)
val_ds = val_ds.batch(batch_size)



input_shape = (49, 40, 1)
num_labels = len(a.commands)  # will include silence/unknown

assert num_labels == model_settings["label_count"]

# NEW MODEL
#
# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#b0-to-b7-variants-of-efficientnet
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights=None,  # "imagenet",
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    # classes=1000,
    # classifier_activation="softmax",
)

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation="relu")(x)
# layers.Dropout(0.5)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dense(192, kernel_initializer="lecun_normal", activation="selu")(x)
# must use alpha-dropout if dropout is desired with selu
logits = layers.Dense(num_labels)(x)

model = models.Model(inputs=base_model.input, outputs=logits)

model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# TODO(mmaz) class_weight parameter on model.fit

# LOAD PREVIOUS CHECKPOINT
# model_dir = Path(f"/home/mark/tinyspeech_harvard/train_{LANG_ISOCODE}_165/models")
# checkpoint = model_dir / "nl_165commands_efficientnet_selu_specaug80_resume62.008-0.7926"
# model = models.load_model(checkpoint)
# # change learning rate:
# model.compile(
#    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), #<-- change learning rate!
#    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#    metrics=["accuracy"],
# )

# CHANGE FILENAME
EPOCHS = 40
os.chdir(f"/home/mark/tinyspeech_harvard/train_{LANG_ISOCODE}_165/models/")
checkpoint_filepath = (
    LANG_ISOCODE
    + "_165commands_efficientnet_selu_specaug80_resume62_resume08.{epoch:03d}-{val_accuracy:.4f}"
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[model_checkpoint_callback]
    # callbacks=[tf.keras.callbacks.EarlyStopping(verbose=1, patience=4),
    #            tf.keras.callbacks.LearningRateScheduler(scheduler)],
)
history_idx = 0
while os.path.isfile(f"./history_keras_{history_idx}.pkl"):
    history_idx += 1
with open(f"./history_keras_{history_idx}.pkl", "wb") as fh:
    pickle.dump(history.history, fh)
