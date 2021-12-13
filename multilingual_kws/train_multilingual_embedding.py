import os
import pickle

import logging
import glob
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

import sys

import input_data

from pathlib import Path
import pickle

embedding_model_dir = Path("/home/mark/tinyspeech_harvard/multilingual_embedding_wc")
save_models_dir = embedding_model_dir / "models"
os.chdir(embedding_model_dir)

if not os.path.isdir(save_models_dir):
    raise ValueError("create model dir")

# copied from /media/mark/hyperion/multilingual_embedding_data_w_context
with open("commands.txt", "r") as fh:
    commands = fh.read().splitlines()
with open("train_files.txt", "r") as fh:
    train_files = fh.read().splitlines()
with open("val_files.txt", "r") as fh:
    val_files = fh.read().splitlines()

model_settings = input_data.standard_microspeech_model_settings(label_count=761)
bg_datadir = embedding_model_dir / "_background_noise_"

if not os.path.isdir(bg_datadir):
    raise ValueError("no bg data at", bg_datadir)

a = input_data.AudioDataset(
    model_settings,
    commands,
    bg_datadir,
    [],
    silence_percentage=1,
    unknown_percentage=0,
    spec_aug_params=input_data.SpecAugParams(percentage=80),
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = a.init_from_parent_dir(AUTOTUNE, train_files, is_training=True)
val_ds = a.init_from_parent_dir(AUTOTUNE, val_files, is_training=False)
batch_size = 64
train_ds = train_ds.shuffle(buffer_size=8000).batch(batch_size)
val_ds = val_ds.batch(batch_size)


input_shape = (49, 40, 1)
num_labels = len(a.commands)  # will include silence/unknown

assert num_labels == model_settings["label_count"]

# NEW MODEL
#
# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#b0-to-b7-variants-of-efficientnet
# base_model = tf.keras.applications.EfficientNetB0(
#     include_top=False,
#     weights=None,  # "imagenet",
#     input_tensor=None,
#     input_shape=input_shape,
#     pooling=None,
#     # classes=1000,
#     # classifier_activation="softmax",
# )
# x = base_model.output
# x = layers.GlobalAveragePooling2D()(x)
# # x = layers.BatchNormalization()(x)
# x = layers.Dense(2048, activation="relu")(x)
# # layers.Dropout(0.5)
# x = layers.Dense(2048, activation="relu")(x)
# x = layers.Dense(1024, kernel_initializer="lecun_normal", activation="selu")(x)
# # must use alpha-dropout if dropout is desired with selu
# logits = layers.Dense(num_labels)(x)

# model = models.Model(inputs=base_model.input, outputs=logits)
# model.summary()
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )

# TODO(mmaz) class_weight parameter on model.fit

# LOAD PREVIOUS CHECKPOINT
checkpoint = save_models_dir / "multilingual_context_.020-0.7058"
model = models.load_model(checkpoint)
# # change learning rate:
model.compile(
   #optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), #<-- change learning rate!
   optimizer=tf.keras.optimizers.Adam(), #<-- change learning rate!
   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
   metrics=["accuracy"],
)

# CHANGE FILENAME
EPOCHS = 8
os.chdir(save_models_dir)
basename="multilingual_context_resume20_"
checkpoint_filepath = basename + ".{epoch:03d}-{val_accuracy:.4f}"

log_idx = 0
while os.path.isfile(f"{basename}_log_{log_idx}.csv"):
    log_idx += 1
csvlog_dest = f"{basename}_log_{log_idx}.csv"

csvlogger = tf.keras.callbacks.CSVLogger(csvlog_dest, append=False)

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
    callbacks=[csvlogger, model_checkpoint_callback]
    # callbacks=[tf.keras.callbacks.EarlyStopping(verbose=1, patience=4),
    #            tf.keras.callbacks.LearningRateScheduler(scheduler)],
)
history_idx = 0
while os.path.isfile(f"./history_keras_{history_idx}.pkl"):
    history_idx += 1
with open(f"./history_keras_{history_idx}.pkl", "wb") as fh:
    pickle.dump(history.history, fh)
