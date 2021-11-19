# %%
from pathlib import Path
import os
import shutil
import numpy as np
import json
import csv

from inspect import cleandoc

# %%
swts_mp3s_to_filter = "mozilla_swts_en.txt"
with open(swts_mp3s_to_filter, "r") as fh:
    mp3s = fh.read().split()
len(mp3s)

# %%


def generate_microset(
    words, splits_csv, swts_mp3s_to_filter: Path, max_clips_per_kw=6000
):
    """
    Args:
      swts_mp3s_to_filter: filepath to a list of MP3s contained in Mozilla's
        Single Word Target Segments dataset, which are manually recorded individual
        words for [digits 0-9, yes, no, hey, Firefox] - we filter these words because
        they are not extracted from full Common Voice sentences, and thus are not
        representative of the keywords found in the rest of the dataset.

        These files were generated via the following commands:
          cv-corpus-7.0-singleword/en/clips > mozilla_swts_en.txt
          cv-corpus-7.0-singleword/es/clips > mozilla_swts_es.txt
          
    """
    with open(swts_mp3s_to_filter, "r") as fh:
        mp3s = fh.read().split()
    swts_clips = set([Path(mp3).stem for mp3 in mp3s])

    all_samples = dict(train={}, dev={}, test={})
    clip2row = {}
    with open(splits_csv, "r") as fh:
        reader = csv.reader(fh)
        next(reader)  # SET,LINK,WORD,VALID,SPEAKER,GENDER
        for row in reader:
            word = row[2]
            if word in words:
                split = row[0].lower()
                clip = row[1]
                stem = Path(clip).stem
                if stem in swts_clips:
                    continue
                if word not in all_samples[split]:
                    all_samples[split][word] = []
                all_samples[split][word].append(clip)
                clip2row[clip] = row

    # deterministic via ordering in words and en_splits.csv rows
    rng = np.random.RandomState(0)

    micro_dataset = {}
    for split, kws_clips in all_samples.items():
        micro_dataset[split] = {}
        for keyword, clips in kws_clips.items():
            if len(clips) > max_clips_per_kw:
                micro_dataset[split][keyword] = rng.choice(
                    clips, max_clips_per_kw, replace=False
                )
            else:
                micro_dataset[split][keyword] = clips

    # for constructing a CSV of the microset
    selected_rows = []
    for split in ["train", "dev", "test"]:
        for w, clips in micro_dataset[split].items():
            for clip in clips:
                selected_rows.append(clip2row[clip])

    return micro_dataset, selected_rows


def show_counts(microset, rows):
    total = 0
    for split in ["train", "dev", "test"]:
        print(f"---{split}----")
        for w, clips in microset[split].items():
            print(w, len(clips))
            total += len(clips)
    print("total", total)
    assert total == len(rows)


def populate_microset(
    microset,
    rows,
    src_dir: Path,
    max_clips_per_kw,
    language,
    lang_isocode,
    dest_dir: Path,
):
    assert dest_dir.is_dir() and len(list(dest_dir.iterdir())) == 0
    os.makedirs(dest_dir / "clips")
    for split in ["train", "dev", "test"]:
        for word, clips in microset[split].items():
            dest_path = dest_dir / "clips" / word
            os.makedirs(dest_path, exist_ok=True)
            for clip in clips:
                src_file = src_dir / "clips" / clip
                shutil.copy2(src_file, dest_path)
    with open(dest_dir / f"{lang_isocode}_splits.csv", "w") as fh:
        writer = csv.writer(fh)
        header = "SET,LINK,WORD,VALID,SPEAKER,GENDER".split(",")
        writer.writerow(header)
        writer.writerows(rows)
    for split in ["TRAIN", "DEV", "TEST"]:
        with open(dest_dir / f"{lang_isocode}_{split.lower()}.csv", "w") as fh:
            writer = csv.writer(fh)
            header = "LINK,WORD,VALID,SPEAKER,GENDER".split(",")
            writer.writerow(header)
            rows_for_split = [row[1:] for row in rows if row[0] == split]
            writer.writerows(rows_for_split)
    with open(dest_dir / "mswc_microset_readme.txt", "w") as fh:
        info = f"""
        This is a small portion (a "microset") of the data available for the {language} 
        language in the Multilingual Spoken Words Corpus (MSWC), limited to {max_clips_per_kw} 
        clips per keyword, and constrained to only {len(microset["train"])} keywords. The
        intent of this small subset is to aid in preliminary experimentation, inspection,
        and tutorials, without requiring users to download the full MSWC dataset or
        the full subset of MSWC in {language}. To download the full dataset, please visit
        the URL below:

        version 1.0, Multilingual Spoken Words Corpus, https://mlcommons.org/en/multilingual-spoken-words

        """
        fh.write(cleandoc(info) + "\n")


# %%

# words - based on speech_commands v0.2.0, eliding two-letter words
# fmt: off
words_en = { "left", "right", "forward", "backward", "follow", "down", "yes",
    "learn", "house", "bed", "dog", "bird", "cat", "tree", "visual", "wow",
    "happy", "marvin", "sheila", "stop", "off", "zero", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", }

print(len(words_en))
# fmt: on
splits_en = Path("/mnt/disks/std3/keith/splits/output_v2/en_splits.csv")

micro_en, rows_en = generate_microset(
    words_en, splits_en, swts_mp3s_to_filter=Path("mozilla_swts_en.txt")
)

show_counts(micro_en, rows_en)

# %%
populate_microset(
    micro_en,
    rows_en,
    src_dir=Path("/mnt/disks/std3/opus/generated/common_voice/frequent_words/en"),
    max_clips_per_kw=6000,
    language="English",
    lang_isocode="en",
    dest_dir=Path("/mnt/disks/std750/mark/mswc_microset/en"),
)

# %%
words_es = "encuentra ciudad nombre número universidad tiempo vida color canción juego cero uno dos tres cuatro cinco seis siete ocho nueve".split()
splits_es = Path("/mnt/disks/std3/keith/splits/output_v2/es_splits.csv")
micro_es, rows_es = generate_microset(
    words_es, splits_es, swts_mp3s_to_filter=Path("mozilla_swts_es.txt")
)

show_counts(micro_es, rows_es)

# %%
populate_microset(
    micro_es,
    rows_es,
    src_dir=Path("/mnt/disks/std3/opus/generated/common_voice/frequent_words/es"),
    max_clips_per_kw=6000,
    language="Spanish",
    lang_isocode="es",
    dest_dir=Path("/mnt/disks/std750/mark/mswc_microset/es"),
)

# %%
# %%
basedir = Path("/mnt/disks/std750/mark/mswc_microset/es")
splits_csv = basedir / "es_splits.csv"
train_csv = basedir / "es_train.csv"
dev_csv = basedir / "es_dev.csv"
test_csv = basedir / "es_test.csv"
# basedir = Path("/mnt/disks/std750/mark/mswc_microset/en")
# splits_csv = basedir / "en_splits.csv"
# train_csv = basedir / "en_train.csv"
# dev_csv = basedir / "en_dev.csv"
# test_csv = basedir / "en_test.csv"

splits = dict(train=[], dev=[], test=[])
with open(splits_csv, "r") as fh:
    reader = csv.reader(fh)
    next(reader)
    for row in reader:
        splits[row[0].lower()].append(row[1])

paths = {}
for k, f in zip(["train", "dev", "test"], [train_csv, dev_csv, test_csv]):
    with open(f, "r") as fh:
        reader = csv.reader(fh)
        next(reader)
        paths[k] = [row[0] for row in reader]

assert set(splits["train"]) == set(paths["train"])
assert set(splits["dev"]) == set(paths["dev"])
assert set(splits["test"]) == set(paths["test"])

# %%
