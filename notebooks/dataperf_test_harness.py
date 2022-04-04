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
import sklearn.pipeline
import sklearn.linear_model
import sklearn.ensemble
import sklearn.preprocessing

# from sklearn.metrics import classification_report
import sklearn.svm
import sklearn.model_selection

# %%
j = Path.home() / "tinyspeech_harvard/dataperf/metadata.json.gz"
with gzip.open(j, "r") as fh:
    meta = json.load(fh)

# %%


@dataclass
class TestParams:
    minimum_total_samples: int = 500  # for candidate words
    language_isocode: str = "en"
    num_targets: int = 5
    num_experiments: int = 200
    num_splits_per_experiment: int = 10
    # max_num_samples_for_selection: int = 300 # TODO(mmaz) (how) should we enforce this?
    num_target_samples: int = 100
    num_nontarget_training_samples: int = 20
    num_nontarget_eval_samples: int = 200  # ideally - TODO(mmaz) we are missing some unknown samples

    SEED_EXPERIMENT_GENERATION: int = 0
    SEED_NONTARGET_SELECTION: int = 0
    SEED_SPLITTER: int = 0


# TODO(mmaz) should we enforce this?
# assert (
#     TestParams.num_target_samples + TestParams.num_nontarget_training_samples
#     == TestParams.max_num_samples_for_selection
# ), "nontarget and target samples must sum to max samples"

# %%
candidate_words = []
for w, c in meta[TestParams.language_isocode]["wordcounts"].items():
    if c > TestParams.minimum_total_samples:
        candidate_words.append(w)
print(len(candidate_words))

# %%
candidate_words
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


def get_unknown_fvs():
    # use the same unknowns for all folds for now
    # TODO(mmaz) improve this
    unknown_rng = np.random.RandomState(TestParams.SEED_NONTARGET_SELECTION)
    selected_unknowns = unknown_rng.choice(
        unknown_en,
        TestParams.num_nontarget_training_samples
        + TestParams.num_nontarget_eval_samples,
        replace=False,
    )
    missing_unknowns = 0
    selected_fvs = []
    for u in selected_unknowns:
        u_path = Path(u)
        word = u_path.parts[-2]
        df = read_parquet(word)
        parquet_idx = f"{word}/{u_path.stem}.wav"
        fv = df.loc[df["clip_id"] == parquet_idx]
        if fv.shape[0] == 0:
            missing_unknowns += 1
            continue
        fv = np.stack(fv.mswc_embedding_vector.values)
        selected_fvs.append(fv)
    print("WARNING: missing unknowns", missing_unknowns)
    selected_fvs = np.concatenate(selected_fvs)
    print(selected_fvs.shape)
    training_unknowns = selected_fvs[: TestParams.num_nontarget_training_samples, :]
    eval_unknowns = selected_fvs[TestParams.num_nontarget_training_samples :, :]
    print("training_unknowns", training_unknowns.shape)
    print("eval unknowns", eval_unknowns.shape)
    return dict(train=training_unknowns, eval=eval_unknowns)


unknown_fvs = get_unknown_fvs()

# %%
# classifiers


def simplest_svm(train_Xs, train_ys):
    # TODO(mmaz)
    #  * what kind of SVM do we want
    #  * should we use the pipeline for scaling features?
    #  https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    clf = sklearn.svm.SVC()
    clf.fit(train_Xs, train_ys)
    return clf


def svm_with_scaling(train_Xs, train_ys):
    clf = sklearn.pipeline.Pipeline(
        [("scaler", sklearn.preprocessing.StandardScaler()), ("svm", sklearn.svm.SVC())]
    )
    clf.fit(train_Xs, train_ys)
    return clf


def logistic_regression(train_Xs, train_ys):
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(train_Xs, train_ys)
    return clf


def logistic_regression_with_scaling(train_Xs, train_ys):
    clf = sklearn.pipeline.Pipeline(
        [
            ("scaler", sklearn.preprocessing.StandardScaler()),
            ("lr", sklearn.linear_model.LogisticRegression()),
        ]
    )
    clf.fit(train_Xs, train_ys)
    return clf


def random_forest(train_Xs, train_ys):
    clf = sklearn.ensemble.RandomForestClassifier()
    clf.fit(train_Xs, train_ys)
    return clf


# def xgboost(train_Xs, train_ys):
#     clf = sklearn.ensemble.XGBClassifier()
#     clf.fit(train_Xs, train_ys)
#     return clf


def gradient_boosting(train_Xs, train_ys):
    clf = sklearn.ensemble.GradientBoostingClassifier()
    clf.fit(train_Xs, train_ys)
    return clf


def logistic_regression_with_l1_regularization(train_Xs, train_ys):
    clf = sklearn.linear_model.LogisticRegression(penalty="l1", solver="liblinear")
    clf.fit(train_Xs, train_ys)
    return clf


def logistic_regression_with_l2_regularization(train_Xs, train_ys):
    clf = sklearn.linear_model.LogisticRegression(penalty="l2")
    clf.fit(train_Xs, train_ys)
    return clf


# def logistic_regression_with_elastic_net_regularization(train_Xs, train_ys):
#     clf = sklearn.linear_model.LogisticRegression(penalty="elasticnet")
#     clf.fit(train_Xs, train_ys)
#     return clf

# def logistic_regression_with_lasso_regularization(train_Xs, train_ys):
#     clf = sklearn.linear_model.LogisticRegression(penalty="lasso")
#     clf.fit(train_Xs, train_ys)
#     return clf

# def logistic_regression_with_ridge_regularization(train_Xs, train_ys):
#     clf = sklearn.linear_model.LogisticRegression(penalty="ridge")
#     clf.fit(train_Xs, train_ys)
#     return clf

classifier_functions = [
    simplest_svm,
    svm_with_scaling,
    logistic_regression,
    logistic_regression_with_scaling,
    random_forest,
    gradient_boosting,
    logistic_regression_with_l1_regularization,
    logistic_regression_with_l2_regularization,
]

# %%


def experiment_run(e):
    # reserve index 0 for unknown
    word2classid = {word: ix + 1 for ix, word in enumerate(e)}

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

    # add unknowns to evals (for now we add unknowns to the training samples within each fold)
    eval_samples = np.vstack([eval_samples, unknown_fvs["eval"]])
    eval_labels = np.concatenate([eval_labels, [0] * unknown_fvs["eval"].shape[0]])

    selector = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=TestParams.num_splits_per_experiment,
        train_size=TestParams.num_target_samples,
        random_state=TestParams.SEED_SPLITTER,
    )
    # cross-validation on training data
    classifiers = []
    crossfold_scores = []
    for train_ixs, val_ixs in selector.split(training_samples, training_labels):
        train_Xs = training_samples[train_ixs]
        train_ys = training_labels[train_ixs]
        # TODO(mmaz) look at the distribution of unknowns vs the target samples

        # add unknowns
        train_Xs = np.vstack([train_Xs, unknown_fvs["train"]])
        train_ys = np.concatenate([train_ys, [0] * unknown_fvs["train"].shape[0]])

        val_Xs = training_samples[val_ixs]
        val_ys = training_labels[val_ixs]

        # add unknowns to val fold (these are the same as in the final eval split for now, TODO(mmaz))
        val_Xs = np.vstack([val_Xs, unknown_fvs["eval"]])
        val_ys = np.concatenate([val_ys, [0] * unknown_fvs["eval"].shape[0]])

        # classifiers_per_fold = []
        # scores_per_fold = []
        # for cfxn in classifier_functions:
        #     clf = cfxn(train_Xs, train_ys)
        #     scores_per_fold.append(clf.score(val_Xs, val_ys))
        #     classifiers_per_fold.append(clf)
        # crossfold_scores.append(scores_per_fold)
        # classifiers.append(classifiers_per_fold)

        clf = sklearn.ensemble.VotingClassifier(
            estimators=[
                ("svm", sklearn.svm.SVC(probability=True)),
                ("lr", sklearn.linear_model.LogisticRegression()),
            ],
            voting="soft",
            weights=None,
        )
        clf.fit(train_Xs, train_ys)
        classifiers.append(clf)

        score = clf.score(val_Xs, val_ys)
        crossfold_scores.append(score)

    # crossfold_scores = np.array(crossfold_scores)
    # best_clf_ix = np.unravel_index(np.argmax(crossfold_scores, axis=None), crossfold_scores.shape)
    # best_clf = classifiers[best_clf_ix[0]][best_clf_ix[1]]
    # experiment_score = best_clf.score(eval_samples, eval_labels)
    # experiment_results = dict(words=e, score=experiment_score, best_clf_ix=best_clf_ix)
    best_clf = classifiers[np.argmax(crossfold_scores)]
    experiment_score = best_clf.score(eval_samples, eval_labels)
    experiment_results = dict(words=e, score=experiment_score)
    return experiment_results


# s, yd = experiment_run(experiment_list[0])
# print(s)
# print(len(yd))
# plt.hist(yd)

for ix, e in enumerate(experiment_list):
    print(ix, e)
experiment_results = []
experiment_scores = []
with multiprocessing.Pool(processes=8) as p:
    for r in p.imap_unordered(experiment_run, experiment_list, chunksize=5):
        experiment_scores.append(r["score"])
        experiment_results.append(r)
        if len(experiment_scores) % int(TestParams.num_experiments / 10) == 0:
            print(f"{len(experiment_scores)}/{TestParams.num_experiments} done")

plt.hist(experiment_scores, bins=25)
plt.show()

for r in experiment_results:
    print(r["best_clf_ix"])

# %%
