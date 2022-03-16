# %%
import datetime
import multiprocessing
import os
import json
from pathlib import Path
import numpy as np
import tqdm
import sox
from dataclasses import dataclass
import pickle

# %%
with open("metadata.json", "r") as fh:
    meta = json.load(fh)

@dataclass
class Task:
    clip_dir: Path
    sample_opus: str
    word: str
    basedir: Path

target_dir = Path("/mnt/disks/std4/16khz_wav")

def convert_16k_wav(task:Task):
    wav = Path(task.sample_opus).stem + ".wav"
    original_fp = str(task.basedir / task.word / wav)
    dest = str(task.clip_dir / wav)
    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)
    transformer.build(original_fp, dest)
    return task

#fmt: off
mswc_languages = ['sl', 'br', 'ro', 'rm-sursilv', 'el', 'mt', 'id', 'sah', 'fy-NL', 'cv', 'sk', 'ia', 'lv', 'vi', 'ar', 'as', 'or', 'gn', 'sv-SE', 'dv', 'ta', 'rm-vallader', 'ka', 'zh-CN', 'cnh', 'ha', 'ga-IE', 'ky', 'mn', 'tr', 'lt', 'uk', 'et', 'cs', 'tt', 'pt', 'nl', 'cy', 'ru', 'eo', 'fa', 'eu', 'pl', 'rw', 'ca', 'it', 'de', 'en', 'es', 'fr']
assert len(mswc_languages) == 50


# std3_ignore = ['ab', 'gl']
std3_langs = ['br', 'cv', 'gn', 'de', 'el', 'en', 'fa', 'ca', 'ha', 'lt', 'pt', 'ru', 'rw', 'sk', 'ta', 'tr', 'tt', 'uk', 'zh-CN']
for lang in std3_langs:
    assert lang in mswc_languages, f"{lang} missing"

superset_std2_langs = ['it', 'mt', 'pl', 'sl', 'ka', 'rm-sursilv', 'rm-vallader', 'mn', 'ro', 'es', 'eo', 'sah', 'lv', 'fr', 'eu', 'cnh', 'pa-IN', 'dv', 'ar', 'fi', 'vi', 'as', 'cy', 'lg', 'ab', 'or', 'cs', 'fy-NL', 'sv-SE', 'hi', 'id', 'ky', 'ia', 'et', 'ga-IE', 'ja', 'nl'] # removed 'el'
std2_mswc_langs = list(set(mswc_languages).intersection(superset_std2_langs))
print(std2_mswc_langs)

# isocodes_to_process = std3_langs
isocodes_to_process = std2_mswc_langs

basedir_std3 = "/mnt/disks/std3/data/generated/common_voice/frequent_words/{lang_isocode}/clips"
basedir_std2 = "/mnt/disks/std2/data/generated/common_voice/frequent_words/{lang_isocode}/clips"
basedir_template = basedir_std2
#fmt:on

verify = True
if verify:
    for lang_isocode in isocodes_to_process:
        print(f"verifying all clips in MSWC metadata are present on disk for {lang_isocode}")
        basedir = Path(basedir_template.format(lang_isocode))
        uhoh = 0
        considered = 0
        for keyword, samples in tqdm.tqdm(meta[lang_isocode]["filenames"].items()):
            for sample in samples:
                wav = Path(sample).stem + ".wav"
                fp = basedir / keyword / wav
                if not fp.exists():
                    uhoh += 1
                considered += 1
        print(lang_isocode, considered, uhoh)
        assert uhoh == 0, "missing source wavfiles"


for lang_isocode in isocodes_to_process:
    basedir = Path(basedir_template.format(lang_isocode))
    full_name = meta[lang_isocode]["language"]
    print(f"\n\n starting {lang_isocode} -------- {full_name}")

    tasks = []
    keywords = meta[lang_isocode]["wordcounts"].keys()
    for word in keywords:
        clip_dir = target_dir / lang_isocode / "clips" / word
        os.makedirs(clip_dir, exist_ok=True)
        for sample_opus in meta[lang_isocode]["filenames"][word]:
            tasks.append(Task(clip_dir=clip_dir, sample_opus=sample_opus, word=word, basedir=basedir))

    total = len(tasks)
    print("tasks:", total)
    five_pct = int(len(tasks) * 0.05)

    done = 0
    with multiprocessing.Pool() as p:
        for i in p.imap_unordered(convert_16k_wav, tasks, chunksize=200):
            done += 1
            if done % five_pct == 0:
                print(f"{lang_isocode} {done=} {total=}", done / total)
    now = datetime.datetime.now()
    with open(f"logs/{lang_isocode}__{now}.log", 'w') as fh:
        fh.write(f"{done}\n")
    print(f"{lang_isocode} all done", done, len(tasks), "pct:", done / len(tasks), now)
# %%