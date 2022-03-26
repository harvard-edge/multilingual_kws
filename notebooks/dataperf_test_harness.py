# %%
from dataclasses import dataclass
import multiprocessing
import csv
import json
import gzip
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.svm
import sklearn.model_selection

# %%
j = Path.home() / "tinyspeech_harvard/dataperf/metadata.json.gz"
with gzip.open(j, "r") as fh:
    meta = json.load(fh)

# %%


@dataclass
class TestParams:
    minimum_total_samples: int = 300  # for candidate words
    language_isocode: str = "en"
    num_targets: int = 10 
    num_experiments: int = 100
    num_splits_per_experiment: int = 10
    max_num_samples_for_selection: int = 500
    num_target_samples: int = 400
    num_nontarget_samples: int = 100
    SEED_EXPERIMENT_GENERATION: int = 0
    SEED_NONTARGET_SELECTION: int = 0
    SEED_SPLITTER: int = 0


assert (
    TestParams.num_target_samples + TestParams.num_nontarget_samples
    == TestParams.max_num_samples_for_selection
), "nontarget and target samples must sum to max samples"

# %%
candidate_words = []
for w, c in meta[TestParams.language_isocode]["wordcounts"].items():
    if c > TestParams.minimum_total_samples:
        candidate_words.append(w)
print(len(candidate_words))

# %%
splits_file = Path("/media/mark/hyperion/mswc/splits_en/en_splits.csv")
assert splits_file.stem.split("_")[0] == TestParams.language_isocode

sample2split = {}  # map of opus sample filenames (from metadata.json) to split
with open(splits_file, "r") as fh:
    reader = csv.reader(fh)
    next(reader)  # SET,LINK,WORD,VALID,SPEAKER,GENDER
    for row in reader:
        clip = row[1]  # aachen/common_voice_en_18833718.opus
        word = row[2]
        split = row[0].lower()
        sample2split[Path(clip).name] = dict(word=word, split=split)
# %%
def train_dev_test(word):
    """
    searches metadata.json for all samples filenames (opus) for a given keyword
    Returns: parquet .wav indices for the requested split
    """
    samples = meta[TestParams.language_isocode]["filenames"][word]
    train, dev, test = [], [], []
    for s in samples:
        # weathering/common_voice_en_18772697.wav
        parquet_index = str(Path(word) / (Path(s).stem + ".wav"))
        split = sample2split[s]["split"]
        if split == "train":
            train.append(parquet_index)
        elif split == "dev":
            dev.append(parquet_index)
        elif split == "test":
            test.append(parquet_index)
    return dict(train=train, dev=dev, test=test)


train_dev_test("weathering")

# %%
# load MSWC embedding's unknown_en split
unknown_files_txt = Path.home() / "tinyspeech_harvard/unknown_files/unknown_files.txt"
unknown_samples_base = Path.home() / "tinyspeech_harvard/unknown_files"
unknown_files = []
unknown_en = []
with open(unknown_files_txt, "r") as fh:
    for w in fh.read().splitlines():
        unknown_files.append(unknown_samples_base / w)
        # print(w)
        lang = Path(w).parts[1]
        # print(lang)
        if lang == "en":
            unknown_en.append(unknown_samples_base / w)
print("Number of unknown files", len(unknown_files))
print("Number of unknown en", len(unknown_en))

unknown_en_words = [Path(w).parts[-2] for w in unknown_en]
unknown_set = set(unknown_en_words)
print(unknown_set)
print(len(unknown_en_words) - len(unknown_set))

print(
    "candidate word overlap?", set(candidate_words).intersection(set(unknown_en_words))
)
# %%
# experiment with model selection strategies
ws = ["weather" + str(i) for i in range(20)]
ds = ["date" + str(i) for i in range(20)]
ts = ["time" + str(i) for i in range(20)]

wdt = ws + ds + ts
wdt_labels = [0 for _ in ws] + [1 for _ in ds] + [2 for _ in ts]

s = sklearn.model_selection.StratifiedShuffleSplit(
    n_splits=6, train_size=7, random_state=0
)
for train, test in s.split(wdt, wdt_labels):
    print("-------")
    print(np.array(wdt)[train], len(test))

# %%
# experiment generation

rng = np.random.RandomState(TestParams.SEED_EXPERIMENT_GENERATION)

experiment_list = []
while len(experiment_list) < TestParams.num_experiments:
    candidate_exp = set(
        rng.choice(candidate_words, TestParams.num_targets, replace=False)
    )
    # in our unknown list?
    if len(candidate_exp.intersection(unknown_set)) > 0:
        continue
    # are any of these words in previously scheduled experiments?
    if any([len(candidate_exp.intersection(e)) > 0 for e in experiment_list]):
        continue
    experiment_list.append(candidate_exp)

for ix, e in enumerate(experiment_list):
    print(ix, e)


# test harness
def read_parquet(word, parquet_basedir=Path("/media/mark/hyperion/mswc/embeddings/en")):
    return pd.read_parquet(parquet_basedir / (word + ".parquet"))


def get_fvs_by_split(word):
    """ look up feature vectors by split
    Returns: dict[str,np.ndarray]
        for example:
          weather: {train: (970, 1024), dev:, test:, ...}
    """
    df = read_parquet(word)
    # a dict of parquet .wav indices, for the keys train, dev, test
    results = {}
    for split, clip_ids in train_dev_test(word).items():
        split_df = df[df["clip_id"].isin(clip_ids)]
        split_fvs = np.stack(split_df.mswc_embedding_vector.values)
        results[split] = split_fvs
    return results


experiment_results = []
experiment_scores = []
def experiment_run(e):
    word2classid = {word: ix for ix, word in enumerate(e)}

    #  {weather : {train: (970, 1024), dev:, test:, ...}}}
    word2splits_fvs = {word: get_fvs_by_split(word) for word in e}

    training_samples = []
    training_labels = []

    eval_samples = []
    eval_labels = []
    for word, splits2fvs in word2splits_fvs.items():
        train_fvs = splits2fvs["train"]
        training_samples.append(train_fvs)
        training_labels.extend([word2classid[word] for _ in range(train_fvs.shape[0])])

        eval_fvs = splits2fvs["dev"]
        eval_samples.append(eval_fvs)
        eval_labels.extend([word2classid[word] for _ in range(eval_fvs.shape[0])])

    training_samples = np.vstack(training_samples)
    eval_samples = np.vstack(eval_samples)

    assert training_samples.shape[0] == len(
        training_labels
    ), "labels and samples must be same length"

    training_labels = np.array(training_labels)
    eval_labels = np.array(eval_labels)

    selector = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=TestParams.num_splits_per_experiment,
        train_size=TestParams.max_num_samples_for_selection,
        random_state=TestParams.SEED_SPLITTER,
    )
    # cross-validation on training data
    classifiers = []
    crossfold_scores = []
    for train_ixs, val_ixs in selector.split(training_samples, training_labels):
        train_Xs = training_samples[train_ixs]
        train_ys = training_labels[train_ixs]

        # TODO(mmaz) 
        #  * what kind of SVM do we want
        #  * should we use the pipeline for scaling features?
        #  https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
        clf = sklearn.svm.SVC()
        clf.fit(train_Xs, train_ys)
        classifiers.append(clf)

        val_Xs = training_samples[val_ixs]
        val_ys = training_labels[val_ixs]

        score = clf.score(val_Xs, val_ys)
        crossfold_scores.append(score)


    best_clf = classifiers[np.argmax(crossfold_scores)]
    experiment_score = best_clf.score(eval_samples, eval_labels)
    experiment_results = dict(words=e, score=experiment_score)
    return experiment_results

with multiprocessing.Pool() as p:
    for r in p.imap_unordered(experiment_run, experiment_list, chunksize=5):
        experiment_scores.append(r["score"])
        experiment_results.append(r)
        if len(experiment_scores) % int(TestParams.num_experiments / 10) == 0:
            print(f"{len(experiment_scores)}/{TestParams.num_experiments} done")

plt.hist(experiment_scores, bins=25)

# %%
