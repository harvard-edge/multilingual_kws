import os
import pickle

import logging
import glob
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

import input_data

from pathlib import Path
import pickle


data_dir = Path("/home/mark/tinyspeech_harvard/frequent_words/rw/clips/")
os.chdir("/home/mark/tinyspeech_harvard/train_rw_165/")

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
bg_datadir = "/home/mark/tinyspeech_harvard/frequent_words/rw/_background_noise_/"

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

#  # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#b0-to-b7-variants-of-efficientnet
#  base_model = tf.keras.applications.EfficientNetB0(
#      include_top=False,
#      weights=None, #"imagenet",
#      input_tensor=None,
#      input_shape=input_shape,
#      pooling=None
#      #classes=1000,
#      #classifier_activation="softmax",
#  )
#
#  x = base_model.output
#  x = layers.GlobalAveragePooling2D()(x)
#  x = layers.Dense(1024, activation='relu')(x)
#  #layers.Dropout(0.5)
#  x = layers.Dense(1024, activation='relu')(x)
#  x = layers.Dense(192, kernel_initializer='lecun_normal', activation='selu')(x)
#  # must use alpha-dropout if dropout is desired with selu
#  logits = layers.Dense(num_labels)(x)
#
#  model = models.Model(inputs=base_model.input, outputs=logits)
#
#  model.summary()

# TODO(mmaz) class_weight parameter on model.fit

model_dir = Path("/home/mark/tinyspeech_harvard/train_rw_165/models")
checkpoint = model_dir / "rw_165commands_efficientnet_selu_specaug80.093-0.7715"
model = models.load_model(checkpoint)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

EPOCHS = 60
os.chdir("/home/mark/tinyspeech_harvard/train_rw_165/models/")
checkpoint_filepath = (
    "rw_165commands_efficientnet_selu_specaug80_resume93.{epoch:03d}-{val_accuracy:.4f}"
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
with open("history_keras.pkl", "wb") as fh:
    pickle.dump(history.history, fh)
