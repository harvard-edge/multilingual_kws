# %%
import os
import glob
import multiprocessing
from pathlib import Path

import sox

# %%

basedir = Path.home() / "tinyspeech_harvard/distance_sorting/cv7_extractions"
original = basedir / "listening_data_48k"
target = basedir / "listening_data"


def convert_16k_wav(word):
    os.makedirs(target / word, exist_ok=True)
    for wav in glob.glob(str(original / word / "*.wav")):
        dest = target / word / Path(wav).name
        transformer = sox.Transformer()
        transformer.convert(samplerate=16000)  # from 48K mp3s
        transformer.build(wav, str(dest))
    return word

with multiprocessing.Pool() as p:
    for i in p.imap_unordered(convert_16k_wav, os.listdir(original)):
        print(i)
print("done")
# %%

