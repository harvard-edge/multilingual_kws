#%%
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
from typing import Set, List, Dict, Set
import functools
from collections import Counter
import csv
import textgrid
import sox
import pickle
from scipy.io import wavfile
#import tensorflow as tf
#import input_data
from pathlib import Path
import pydub
from pydub.playback import play
import time

# %%
iso2lang = {
    "ar": "Arabic",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ky": "Kyrgyz",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "ta": "Tamil",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukranian",
}

# %%
paper_data = Path("/home/mark/tinyspeech_harvard/paper_data")
# in_embedding_mlc_pkl = paper_data / "multilang_classification_in_embedding_all_lang_targets.pkl"
# with open(in_embedding_mlc_pkl, 'rb') as fh:
#     in_embedding_mlc = pickle.load(fh)
# for target_data in in_embedding_mlc:
#     print(target_data.keys())
#     break
data_dir = Path("/home/mark/tinyspeech_harvard/frequent_words")
target_data = []
multilang_results_dir = paper_data / "ooe_multilang_classification"
for multiclass_lang in os.listdir(multilang_results_dir):
    lang_isocode = multiclass_lang.split("_")[-1]
    #print("lang_isocode", lang_isocode)
    for result_file in os.listdir(multilang_results_dir / multiclass_lang / "results"):
        target_word = os.path.splitext(result_file.split("_")[-1])[0]
        d = (lang_isocode, target_word)
        target_data.append(d)

random_ix = np.random.randint(len(target_data))
target_lang, target_word = target_data[random_ix]
print(target_lang, iso2lang[target_lang], target_word)

wavs = glob.glob(str(data_dir / target_lang / "clips" / target_word / "*.wav"))
selected = np.random.choice(wavs, 5, replace=False)
#print(selected)
for w in selected:
    print(w)
    play(pydub.AudioSegment.from_file(w))
    time.sleep(0.5)

# f = np.random.choice(utterances, 1)[0]
# print(f)
# f = "/home/mark/tinyspeech_harvard/frequent_words/rw/clips/umuryango/common_voice_rw_21187284__4.wav"
# print(f)
# sr, data = wavfile.read(f)
# print(sr)
# plt.plot(data)
# play(pydub.AudioSegment(data=data, sample_width=2, frame_rate=sr, channels=1))
#%%

# train_files = "/home/mark/tinyspeech_harvard/train_rw_165/train_files.txt"
# with open(train_files, "r") as fh:
#     utterances = fh.read().splitlines()
# commands = "/home/mark/tinyspeech_harvard/train_rw_165/commands.txt"
# with open(commands, "r") as fh:
#     commands = fh.read().splitlines()

#%%
# model_settings = input_data.standard_microspeech_model_settings(label_count=165)
# bg_datadir = "/home/mark/tinyspeech_harvard/frequent_words/rw/_background_noise_/"
# a = input_data.AudioDataset(
#     model_settings,
#     commands,
#     bg_datadir,
#     [],
#     unknown_percentage=0,
#     spec_aug_params=input_data.SpecAugParams(percentage=80),
# )

#%%
# f = np.random.choice(utterances, 1)[0]
# print(f)
# s = a.file2spec_w_bg(f)
# print(s.shape)
# plt.imshow(s)


#%%
#    f = np.random.choice(utterances, 1)[0]
#    f = "/home/mark/tinyspeech_harvard/frequent_words/rw/clips/umuryango/common_voice_rw_21187284__4.wav"
#    print(f)
#    audio_binary = tf.io.read_file(f)
#    waveform = a.decode_audio(audio_binary)
#    # waveform = a._add_bg(waveform)
#    waveform = waveform.numpy()
#    plt.plot(waveform)
#    print(waveform.shape)
#    sr=16_000
#    # this doesnt work - not handling floating point wavdata for some reason??
#    #https://github.com/jiaaro/pydub/blob/master/API.markdown
#    play(pydub.AudioSegment(data=waveform, sample_width=4, frame_rate=sr, channels=1))


#%%

# f = np.random.choice(utterances, 1)[0]
# print(f)
# f = "/home/mark/tinyspeech_harvard/frequent_words/rw/clips/umuryango/common_voice_rw_21187284__4.wav"
# print(f)
# sr, data = wavfile.read(f)
# print(sr)
# plt.plot(data)
# play(pydub.AudioSegment(data=data, sample_width=2, frame_rate=sr, channels=1))

#%%
# f = np.random.choice(utterances, 1)[0]
# f = "/home/mark/tinyspeech_harvard/frequent_words/rw/clips/umuryango/common_voice_rw_21187284__4.wav"
# print(f)
# audio_binary = tf.io.read_file(f)
# waveform = a.decode_audio(audio_binary)
# waveform = a._add_bg(waveform)
# waveform = waveform.numpy()
# plt.plot(waveform)
# print(waveform.shape)
# sr=16_000
# dest = "/home/mark/tinyspeech_harvard/tmp/scratch_wav_data.wav"
# wavfile.write(dest, sr, waveform)
# # this doesnt work - not handling floating point wavdata for some reason??
# #https://github.com/jiaaro/pydub/blob/master/API.markdown
# play(pydub.AudioSegment.from_file(dest))