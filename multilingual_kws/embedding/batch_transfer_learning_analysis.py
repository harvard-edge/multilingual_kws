#%%
import logging
import os
import re
from typing import Dict, List
import datetime
from dataclasses import asdict, dataclass
from pathlib import Path

import glob
import numpy as np
import pickle
import multiprocessing

import sys

import embedding.input_data as input_data

import embedding.transfer_learning as transfer_learning


#%%


def generate_random_non_target_files_for_eval(
    target_word,
    unknown_lang_words,
    oov_lang_words,  # these should prob be in the language being evaluated
    commands,
    frequent_words,
    n_non_targets_per_category=100,
    n_utterances_per_non_target=80,
):
    # select unknown/oov/command words for model evaluation (**not for finetuning**)

    # 3 categories: unknown, oov, command
    some_unknown_ixs = np.random.choice(
        range(len(unknown_lang_words)), n_non_targets_per_category, replace=False
    )
    some_lang_unknown = np.array(unknown_lang_words)[some_unknown_ixs]
    if len(oov_lang_words) < n_non_targets_per_category:
        print(
            "using all OOV words for evaluation",
            target_word,
            len(oov_lang_words),
            flush=True,
        )
        some_lang_oov = np.array(oov_lang_words)
    else:
        some_oov_ixs = np.random.choice(
            range(len(oov_lang_words)), n_non_targets_per_category, replace=False
        )
        some_lang_oov = np.array(oov_lang_words)[some_oov_ixs]
    some_commands = np.random.choice(
        commands, n_non_targets_per_category, replace=False
    )
    some_lang_commands = []
    for c in some_commands:
        for lang in os.listdir(frequent_words):
            if os.path.isdir(frequent_words / lang / "clips" / c):
                some_lang_commands.append((lang, c))
                break

    # filter out target word if present
    flat = [
        (lang, word)
        for lw in [some_lang_unknown, some_lang_oov, some_lang_commands]
        for lang, word in lw
        if word != target_word
    ]
    non_target_files = []
    for lang, word in flat:
        word_dir = frequent_words / lang / "clips" / word
        wavs = glob.glob(str(word_dir / "*.wav"))
        if len(wavs) > n_utterances_per_non_target:
            sample = np.random.choice(wavs, n_utterances_per_non_target, replace=False)
            non_target_files.extend(sample)
        else:
            non_target_files.extend(wavs)
    return non_target_files


def results_exist(target, results_dir):
    # make sure we are not overwriting existing results
    results = [
        os.path.splitext(r)[0]
        for r in os.listdir(results_dir)
        if os.path.splitext(r)[1] == ".pkl"
    ]
    for r in results:
        result_target = r.split("_")[-1]
        if result_target == target:
            return True
    return False


@dataclass
class TargetData:
    lang_ix: int
    target_ix: int
    target_word: str
    target_lang: str
    train_files: List[str]
    val_files: List[str]
    target_wavs: List[str]
    unknown_files: List[str]
    unknown_sample: List[str]
    oov_lang_words: List
    dest_dir: os.PathLike
    base_model_path: os.PathLike
    base_model_output: str


def run_transfer_learning(td: TargetData):
    import tensorflow as tf

    model_settings = input_data.standard_microspeech_model_settings(label_count=3)

    # fmt: off
    result_pkl = Path(td.dest_dir) / "results" / f"{td.lang_ix:02d}_{td.target_ix:03d}_{td.target_lang}_{td.target_word}.pkl"
    csvlog_dest = str(Path(td.dest_dir) / "models" / f"{td.target_word}_trainlog.csv")
    # fmt: on

    assert not os.path.isfile(csvlog_dest), f"{csvlog_dest} csvlog already exists"
    assert not os.path.exists(result_pkl), f"{result_pkl} exists"

    # TODO(mmaz): use keras class weights?
    name, model, details = transfer_learning.transfer_learn(
        target=td.target_word,
        train_files=td.train_files,
        val_files=td.val_files,
        unknown_files=td.unknown_files,
        num_epochs=4,
        num_batches=1,
        batch_size=64,
        model_settings=model_settings,
        csvlog_dest=csvlog_dest,
        base_model_path=td.base_model_path,
        base_model_output=td.base_model_output,
    )
    # fmt: off
    save_dest = Path(td.dest_dir) / "models" / f"{td.lang_ix:02d}_{td.target_ix:03d}_{td.target_lang}_{td.target_word}__{name}"
    # fmt: on
    print("SAVING", save_dest, flush=True)
    model.save(save_dest)

    target_results, all_preds_target = transfer_learning.evaluate_files_single_target(
        td.target_wavs, 2, model, model_settings
    )
    # note: passing in the _TARGET_ category ID (2) for negative examples too:
    # we ignore other categories altogether
    unknown_results, all_preds_unknown = transfer_learning.evaluate_files_single_target(
        td.unknown_sample, 2, model, model_settings
    )

    results = dict(
        target_results=target_results,
        unknown_results=unknown_results,
        all_predictions_targets=all_preds_target,
        all_predictions_unknown=all_preds_unknown,
        details=details,
        target_word=td.target_word,
        target_lang=td.target_lang,
        train_files=td.train_files,
        oov_words=oov_lang_words,
        commands=commands,
        target_data=asdict(td),
    )

    with open(result_pkl, "wb",) as fh:
        pickle.dump(results, fh)

    # https://keras.io/api/utils/backend_utils/
    tf.keras.backend.clear_session()

    return


#%%

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

###############################################
##        MULTILANG MODEL
###############################################

traindir = Path(f"/home/mark/tinyspeech_harvard/multilang_embedding")
with open(traindir / "commands.txt", "r") as fh:
    commands = fh.read().splitlines()
bg_datadir = "/home/mark/tinyspeech_harvard/speech_commands/_background_noise_/"
if not os.path.isdir(bg_datadir):
    raise ValueError("no bg data", bg_datadir)

frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words/")

base_model_path = (
    traindir / "models" / "multilang_resume40_resume05_resume20.022-0.7969"
)

paper_data = Path("/home/mark/tinyspeech_harvard/paper_data/")
base_result_dir = paper_data / "ooe_multilang_classification"

assert os.listdir(base_result_dir) == [], f"there are already results in {base_result_dir}"

# only sample unknown words from languages already in multilingual embedding model
# otherwise eval would be unfair (finetuned model might see more than just
# the 5shot examples, i.e., unknown words in target language )
unknown_collection_path = paper_data / "base_multilang_unknown_collection.pkl"
with open(unknown_collection_path, "rb") as fh:
    unknown_collection = pickle.load(fh)
unknown_lang_words = unknown_collection["unknown_lang_words"]
unknown_files = unknown_collection["unknown_files"]
commands = unknown_collection["commands"]

unknown_word_set = set([w for l, w in unknown_lang_words])
command_set = set(commands)

#%%

# python embedding/batch_transfer_learning_analysis.py > ~/tinyspeech_harvard/paper_data/ooe_multilang_classification_batch_analysis.log

N_SHOTS = 5
NUM_TARGETS_TO_EVAL = 20  # per language
N_NON_TARGETS_PER_CATEGORY = 100  # unknown, oov, command
N_UTTERANCES_PER_NON_TARGET = 80
MAX_N_TRIES = 100

language_isocodes = os.listdir(frequent_words)

# fmt: off
lang_isos_within_embedding_model = ["en", "fr", "ca", "rw", "de", "it", "nl", "es", "fa"]

# for languages out of embedding
lang_isos_out_of_embedding = ['ar', 'cs', 'cy', 'et', 'eu', 'id', 'ky', 'pl', 'pt', 'ru', 'tr', 'tt', 'uk']
# fmt: on

DRY_RUN = False


# prepare training data for all targets
all_lang_targets = []
for lang_ix, lang_isocode in enumerate(lang_isos_out_of_embedding):
    print(
        f":::::::::::::::::::{lang_ix} - {lang_isocode}::::::::::::::::::::::",
        flush=True,
    )

    # find oov_lang_words for language under test
    # this can continually grow as we add more word extractions to frequent_words
    oov_lang_words = []
    lang_words = os.listdir(frequent_words / lang_isocode / "clips")
    for w in lang_words:
        if w in command_set or w in unknown_word_set:
            continue
        oov_lang_words.append((lang_isocode, w))
    print("num oov lang words", len(oov_lang_words), flush=True)

    # choose targets
    targets = []
    for target_ix in range(NUM_TARGETS_TO_EVAL):
        n_tries = 0
        has_results = True
        # skip existing results
        while has_results and n_tries < MAX_N_TRIES:
            rand_ix = np.random.randint(len(oov_lang_words))
            target_lang, target_word = oov_lang_words[rand_ix]
            if target_lang != lang_isocode:  # generate for a specific language
                continue
            # saved_results = results_exist(target_word, results_dir)
            saved_results = False
            will_generate_results = any([d.target_word == target_word for d in targets])
            has_results = saved_results or will_generate_results
            n_tries += 1

        if n_tries >= MAX_N_TRIES:
            raise ValueError("ran out of options for oov words", lang_isocode)

        print("TARGET:", target_lang, target_word, flush=True)
        target_wavs = glob.glob(
            str(frequent_words / target_lang / "clips" / target_word / "*.wav")
        )
        if len(target_wavs) == 0:
            print(
                str(frequent_words / target_lang / "clips" / target_word / "*.wav"),
                flush=True,
            )
            print(
                "*****\n\n*****bug in word search due to unicode issues in",
                target_word,
                "\n\n",
                flush=True,
            )
            continue

        np.random.shuffle(target_wavs)
        train_files = target_wavs[:N_SHOTS]
        val_files = target_wavs[N_SHOTS:]
        # print("\n".join(train_files))

        num_utterances_for_target = len(target_wavs)
        print(
            "n utterances for target",
            target_word,
            num_utterances_for_target,
            flush=True,
        )

        unknown_sample = generate_random_non_target_files_for_eval(
            target_word=target_word,
            unknown_lang_words=unknown_lang_words,
            oov_lang_words=oov_lang_words,
            commands=commands,
            frequent_words=frequent_words,
            n_non_targets_per_category=N_NON_TARGETS_PER_CATEGORY,
            n_utterances_per_non_target=N_UTTERANCES_PER_NON_TARGET,
        )
        # fmt: off
        model_dest_dir = base_result_dir / f"multilang_{lang_isocode}"
        # fmt: on
        d = TargetData(
            lang_ix=lang_ix,
            target_ix=target_ix,
            target_word=target_word,
            target_lang=target_lang,
            train_files=train_files,
            val_files=val_files,
            target_wavs=target_wavs,
            unknown_files=unknown_files,
            unknown_sample=unknown_sample,
            oov_lang_words=oov_lang_words,
            dest_dir=model_dest_dir,
            base_model_path=base_model_path,
            base_model_output="dense_2",
        )
        targets.append(d)
    print("targets generated", len(targets), flush=True)
    if len(targets) < NUM_TARGETS_TO_EVAL:
        print("\n\n\n --- LIMITED DATA --- ", flush=True)
    all_lang_targets.extend(targets)

# save training data
# fmt: off
if not DRY_RUN:
    all_lang_targets_file = "/home/mark/tinyspeech_harvard/paper_data/ooe_multilang_classification_all_lang_targets.pkl"
    assert not os.path.exists(all_lang_targets_file), f"{all_lang_targets_file} already exists"
    with open(all_lang_targets_file, "wb") as fh:
        pickle.dump(all_lang_targets, fh)
# fmt: on

# reorder to viz some data for each language in incremental graph script
np.random.shuffle(all_lang_targets)

# train it!
n_targets = len(all_lang_targets)
for train_ix, d in enumerate(all_lang_targets):
    print(
        f"\n\n\n::::::::::::::::: {train_ix} / {n_targets} ::::{d.target_lang} - {d.target_word} ::::: ",
        flush=True,
    )
    if DRY_RUN:
        continue

    os.makedirs(d.dest_dir, exist_ok=True)
    os.makedirs(d.dest_dir / "models", exist_ok=True)
    os.makedirs(d.dest_dir / "results", exist_ok=True)

    start_clock = datetime.datetime.now()

    p = multiprocessing.Process(target=run_transfer_learning, args=(d,))
    p.start()
    p.join()
    # p.close()
    end_clock = datetime.datetime.now()
    print("elapsed time", end_clock - start_clock, flush=True)

#%%
