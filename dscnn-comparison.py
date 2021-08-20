#%%
import os
from pathlib import Path
import pickle
import logging
import glob
import csv
# import time
# import shutil
# import pprint

import numpy as np
import tensorflow as tf
import sklearn.cluster
import matplotlib.pyplot as plt
import seaborn as sns

import hashlib
import re
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.util import compat

# import pydub
import pydub.playback
import pydub.effects

# import sys

import input_data
import embedding.transfer_learning as tl
import embedding.distance_filtering as ef
from tensorflow.keras import layers

import random

random.seed(10)


sns.set()
sns.set_style("white")
sns.set_palette("bright")


#%%
def model_def(label_count):
    input_shape = (49, 10, 1)
    filters = 64
    weight_decay = 1e-4
    regularizer = tf.keras.regularizers.l2(weight_decay)
    final_pool_size = (int(input_shape[0]/2), int(input_shape[1]/2))
    
    # Model layers
    # Input pure conv2d
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(rate=0.2)(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second layer of separable depthwise conv2d
    x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Third layer of separable depthwise conv2d
    x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Fourth layer of separable depthwise conv2d
    x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Reduce size and apply final softmax
    x = layers.Dropout(rate=0.4)(x)

    x = layers.AveragePooling2D(pool_size=final_pool_size)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(label_count, activation='softmax')(x)

    # Instantiate model.
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

# %%
# def model_def(label_count):
#     input_shape = (49, 10, 1)
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=input_shape),
#         tf.keras.layers.Dense(256, activation='relu'),
#         # tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(256, activation='relu'),
#         # tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(256, activation='relu'),
#         # tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(label_count, activation="softmax")
#     ])

#     return model

# %%
def step_function_wrapper(batch_size):
    def step_function(epoch, lr):
        if (epoch < 12):
            return 0.0005
        elif (epoch < 24):
            return 0.0001
        elif (epoch < 36):
            return 0.00002
        else:
            return 0.00001
    return step_function

# %%
def get_callbacks():
    lr_sched_name ="step_function"
    batch_size = 100
    initial_lr = 0.00001
    callbacks = None
    if(lr_sched_name == "step_function"):
        callbacks = [tf.keras.callbacks.LearningRateScheduler(step_function_wrapper(batch_size),verbose=1)]
    return callbacks

#%%
def train(commands, train_files, val_files, unknown_files, model_settings):
    model = model_def(model_settings['label_count'])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    bg_datadir = Path.home() / "multilingual_kws_dataset/frequent_words/en/_background_noise_"

    if not os.path.isdir(bg_datadir):
        raise ValueError("no bg data at", bg_datadir)

    a = input_data.AudioDataset(
        model_settings,
        commands,
        bg_datadir,
        unknown_files,
        silence_percentage=33.3, #100/12 for equal distribution
        unknown_percentage=33.3,
        spec_aug_params=input_data.SpecAugParams(percentage=80),
    )
    

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = a.init_from_parent_dir(AUTOTUNE, train_files, is_training=True)
    val_ds = a.init_from_parent_dir(AUTOTUNE, val_files, is_training=True)

    batch_size = 100
    train_ds = train_ds.shuffle(buffer_size=len(train_files)).batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    print(np.concatenate([y for x, y in train_ds], axis=0))

    callbacks = [get_callbacks(), tf.keras.callbacks.TensorBoard(log_dir=str(Path.home() / 'multilingual_kws/gsc_log'))]

    model.fit(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=48, callbacks=callbacks)

    return model


# %%

# commands =["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
commands =['right']

current_dir = Path.cwd()

num_classes = len(commands)+2

dataset_dir = current_dir.parent / 'speech_commands'

# model_settings = input_data.standard_microspeech_model_settings(label_count=num_classes) 
model_settings = input_data.prepare_model_settings(
        label_count=num_classes,
        sample_rate=16000,
        clip_duration_ms=1000,
        window_size_ms=30,
        window_stride_ms=20,
        feature_bin_count=10,
        preprocess="micro",
    )
bg_datadir = dataset_dir / "_background_noise_"
print(bg_datadir)

# %%

gsc_train_target_files = []
gsc_train_other_files = []

gsc_val_target_files = []
gsc_val_other_files = []

gsc_test_target_files = []
gsc_test_other_files = []


with open(dataset_dir / 'validation_list.txt','r') as file:
    for line in file.read().splitlines():
        if str(Path(line).parent) in commands:
            gsc_val_target_files.append(str(dataset_dir /  line))
        else:
            gsc_val_other_files.append(str(dataset_dir / line))
with open(dataset_dir / 'testing_list.txt','r') as file:
    for line in file.read().splitlines():
        if str(Path(line).parent) in commands:
            gsc_test_target_files.append(str(dataset_dir /  line))
        else:
            gsc_test_other_files.append(str(dataset_dir / line))  

for path in dataset_dir.glob("**/*"):
    if path.is_file():
        if 'wav' not in path.suffix:
            continue
        if str(path) not in gsc_val_target_files + gsc_val_other_files + gsc_test_target_files + gsc_test_other_files:
            if path.parent.stem in commands:
                gsc_train_target_files.append(str(path))
            else:
                gsc_train_other_files.append(str(path))

random.shuffle(gsc_train_target_files)
random.shuffle(gsc_train_other_files)

# %%
training_file_count = 0
for file in gsc_train_target_files:
    if commands[0] in file:
        training_file_count +=1
print(commands[0]+" train files: " + str(training_file_count))


# %%
clips_dir = current_dir.parent / 'multilingual_kws_dataset/frequent_words/en/clips'
cv_other_files = []
cv_target_clips = {}

for word_dir in clips_dir.glob('*'):
    if not word_dir.name in commands:
        for clip in word_dir.glob('**/*.wav'):
            cv_other_files.append(str(clip))
    else:
        cv_target_clips[word_dir.name] = []
        for clip in word_dir.glob('**/*.wav'):
            cv_target_clips[word_dir.name].append(str(clip))

train_split = 0.8
val_split = 0.1
test_split = 1.0-(train_split+val_split)

cv_train_files = []
cv_val_files = []
cv_test_files = []

for target in cv_target_clips.keys():
    print(target)
    target_list = cv_target_clips[target]
    length = len(target_list)
    print(length)
    train_len = int(length*train_split)
    val_len = int(length*val_split)
    print('val length: ', str(val_len))
    random.shuffle(target_list)

    cv_train_files.extend(target_list[:train_len])
    cv_val_files.extend(target_list[train_len:train_len+val_len])
    cv_test_files.extend(target_list[train_len+val_len:])

random.shuffle(cv_train_files)

# %%
gsc_model = train(commands, gsc_train_target_files,
                gsc_val_target_files, gsc_train_other_files+gsc_val_other_files,
                model_settings)

# %%
cv_train_other_files = cv_other_files[:int(len(cv_other_files)*0.9)]
cv_test_other_files = cv_other_files[len(cv_train_other_files):]
random.shuffle(cv_train_other_files)

# %%
cv_model = train(commands, cv_train_files,
                cv_val_files, cv_train_other_files,
                model_settings)
# %%
a = input_data.AudioDataset(
        model_settings,
        commands,
        bg_datadir,
        gsc_test_other_files,
        silence_percentage=33.3, #100/12 for equal distribution
        unknown_percentage=33.3,
        spec_aug_params=input_data.SpecAugParams(percentage=80),
    )

AUTOTUNE = tf.data.experimental.AUTOTUNE
gsc_test_ds = a.init_from_parent_dir(AUTOTUNE, gsc_test_target_files, is_training=True)
gsc_test_ds = gsc_test_ds.batch(100)
# print(np.concatenate([y for x, y in gsc_test_ds], axis=0))

gsc_test_scores = gsc_model.evaluate(gsc_test_ds)
print("gsc->gsc Test loss:", gsc_test_scores[0])
print("gsc->gsc Test accuracy:", gsc_test_scores[1])

# %%
cv_test_scores = cv_model.evaluate(gsc_test_ds)
print("cv->gsc Test loss:", cv_test_scores[0])
print("cv->gsc Test accuracy:", cv_test_scores[1])
# %%
a = input_data.AudioDataset(
        model_settings,
        commands,
        bg_datadir,
        cv_test_other_files,
        silence_percentage=33.3, #100/12 for equal distribution
        unknown_percentage=33.3,
        spec_aug_params=input_data.SpecAugParams(percentage=80),
    )

AUTOTUNE = tf.data.experimental.AUTOTUNE
cv_test_ds = a.init_from_parent_dir(AUTOTUNE, cv_test_files, is_training=True)
cv_test_ds = cv_test_ds.batch(100)

# print(np.concatenate([y for x, y in cv_test_ds], axis=0))
# print(len(cv_test_other_files))

gsc_test_scores = gsc_model.evaluate(cv_test_ds)
print("gsc->cv Test loss:", gsc_test_scores[0])
print("gsc->cv Test accuracy:", gsc_test_scores[1])

# %%
cv_test_scores = cv_model.evaluate(cv_test_ds)
print("cv->cv Test loss:", cv_test_scores[0])
print("cv->cv Test accuracy:", cv_test_scores[1])
# %%

# %%
