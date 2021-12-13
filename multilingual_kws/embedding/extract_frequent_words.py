import numpy as np
import os
import pandas as pd
import subprocess
import multiprocessing
import csv
from pathlib import Path
import sox
import glob

# local
# ['ar', 'cs', 'cy', 'et', 'eu', 'id', 'ky', 'pl', 'pt', 'ru', 'ta', 'tr', 'tt', 'uk']
LANG_ISOCODE="ky"
WORD_CSVS = f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/timings/*.csv"
#CV_CLIPS_DIR = Path(f"/media/mark/hyperion/common_voice/cv-corpus-5.1-2020-06-22/{LANG_ISOCODE}/clips/")
CV_CLIPS_DIR = Path(f"/media/mark/hyperion/common_voice/cv-corpus-6.1-2020-12-11/{LANG_ISOCODE}/clips/")
#SWTS_CLIPS_DIR = Path("/home/mark/tinyspeech_harvard/commonvoice_singleword/cv-corpus-5-singleword/en/clips")
OUT_DIR = Path(f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/clips")
ERRORS_DIR = Path(f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/errors")
WRITE_PROGRESS = False

# fasrc
# WORD_CSVS = "/n/holyscratch01/janapa_reddi_lab/Lab/mmaz/tinyspeech/frequent_words/en/timings/*.csv"
# CV_CLIPS_DIR = Path("/n/holyscratch01/janapa_reddi_lab/Lab/mmaz/tinyspeech/commonvoice/cv-corpus-5.1-2020-06-22/en/clips")
# SWTS_CLIPS_DIR = Path("/n/holyscratch01/janapa_reddi_lab/Lab/mmaz/tinyspeech/commonvoice_singleword/cv-corpus-5.1-singleword/en/clips")
# OUT_DIR = Path("/n/holyscratch01/janapa_reddi_lab/Lab/mmaz/tinyspeech/frequent_words/en/clips")
# ERRORS_DIR = Path("/n/holyscratch01/janapa_reddi_lab/Lab/mmaz/tinyspeech/frequent_words/en/errors")

def extract_one_second(duration_s: float, start_s: float, end_s: float):
    """
    return one second around the midpoint between start_s and end_s
    """
    if duration_s < 1:
        return (0, duration_s)

    center_s = start_s + ((end_s - start_s) / 2.0)

    new_start_s = center_s - 0.5
    new_end_s = center_s + 0.5

    if new_end_s > duration_s:
        new_end_s = duration_s
        new_start_s = duration_s - 1.0

    if new_start_s < 0:
        new_start_s = 0
        new_end_s = np.minimum(duration_s, new_start_s + 1.0)

#     print(
#         "start",
#         new_start_s,
#         "end",
#         new_end_s,
#         "\nduration",
#         new_end_s - new_start_s,
#         "midpoint",
#         new_start_s + ((new_end_s - new_start_s) / 2.0),
#     )
    return (new_start_s, new_end_s)

def extract(csvpath):
    word = os.path.splitext(os.path.basename(csvpath))[0]
    #print(word)
    if os.path.isdir(OUT_DIR / word):
        raise ValueError("trying to extract to an existing dir", OUT_DIR / word)
    os.mkdir(OUT_DIR / word)

    with open(csvpath, 'r') as fh:
        reader = csv.reader(fh)
        next(reader) # skip header

        for ix, row in enumerate(reader):
            if ix % 1000 == 0:
                print(word, ix)
                if WRITE_PROGRESS:
                    with open("progress.txt", 'a') as fh:
                        fh.write(f"{word} {ix}\n")
            mp3name_no_ext = row[0]
            start_s = float(row[1])
            end_s = float(row[2])
            mp3path = CV_CLIPS_DIR / (mp3name_no_ext + ".mp3")
            #if not os.path.exists(mp3path): # must be in Mozilla SWTS
                #mp3path = SWTS_CLIPS_DIR / (mp3name_no_ext + ".mp3")
            if not os.path.exists(mp3path):
                # really don't know where this came from, skip it
                with open(ERRORS_DIR / mp3name_no_ext, 'a') as fh:
                    pass
                continue

            duration = sox.file_info.duration(mp3path)
            if end_s - start_s < 1:
                pad_amt_s = (1. - (end_s - start_s))/2.
            else: # utterance is already longer than 1s, trim instead
                start_s, end_s = extract_one_second(duration, start_s, end_s)
                pad_amt_s = 0

            dest = OUT_DIR / word / (mp3name_no_ext + ".wav")
            # words can appear multiple times in a sentence: append w number
            count = 2
            while os.path.exists(dest):
                dest = OUT_DIR / word / (f"{mp3name_no_ext}__{count}.wav")
                count += 1
            
            transformer = sox.Transformer()
            transformer.convert(samplerate=16000)  # from 48K mp3s
            transformer.trim(start_s, end_s)
            # use smaller fadein/fadeout since we are capturing just the word
            # TODO(mmaz) is this appropriately sized?
            transformer.fade(fade_in_len=0.025, fade_out_len=0.025)
            transformer.pad(start_duration=pad_amt_s, end_duration=pad_amt_s)
            transformer.build(str(mp3path), str(dest))
    return word

def main():
    if not os.path.isdir(OUT_DIR) or not os.path.isdir(ERRORS_DIR):
        raise ValueError("create outdir and errordir", OUT_DIR, ERRORS_DIR)
    if not os.path.isdir(CV_CLIPS_DIR):
        raise ValueError("data not found")
    all_csvs = glob.glob(WORD_CSVS)
    print("num csvs found", len(all_csvs))
    if len(all_csvs) == 0:
        raise ValueError("no csvs")

    unextracted_csvs = []
    for csvpath in all_csvs:
        word = os.path.splitext(os.path.basename(csvpath))[0]
        wav_dir = OUT_DIR / word
        if os.path.exists(wav_dir) :
            print("skipping", wav_dir)
            continue
        else:
            unextracted_csvs.append(csvpath)
    print("\n\n::::::::::::::::::::")
    print("unextracted csvs:", unextracted_csvs)
    print("\n\n--------------------")
    print("extracting", len(unextracted_csvs), "out of", len(all_csvs))

    pool = multiprocessing.Pool()
    for i, result in enumerate(pool.imap_unordered(extract, unextracted_csvs), start=1):
        print("counter: ", i, "word", result)
        if WRITE_PROGRESS:
            with open("finished.txt", 'a') as fh:
                fh.write(f"counter {i} word {result}\n")
    pool.close()
    pool.join()

    if WRITE_PROGRESS:
        with open("finished.txt", 'a') as fh:
            fh.write("complete\n")


if __name__ == "__main__":
    main()