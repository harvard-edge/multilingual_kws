# %%
from email import header
from pathlib import Path
import pandas as pd
import yaml

# %%
listening_results_basedir = (
    Path.home() / "tinyspeech_harvard/dataperf/listening_results/round1/"
)

experiment_basedir = (
    Path.home() / "tinyspeech_harvard/dataperf/preliminary_evaluation_dataset"
)

list(listening_results_basedir.iterdir())

# %%


def target_validation_filter(target, dryrun=True):
    print("==== validation filter for", target)
    eval_yml_file = experiment_basedir / "eval.yaml"
    eval_yml = yaml.safe_load(eval_yml_file.read_text())

    validation_csv = pd.read_csv(
        listening_results_basedir / f"{target}_validated.csv", header=None
    )

    parquet_file = experiment_basedir / "eval_embeddings" / f"{target}.parquet"

    eval_parquet = pd.read_parquet(parquet_file)

    # verify 1:1 between parquet, yaml, and validation csv
    parquet_clip_ids = set(eval_parquet.clip_id)
    eval_samples = eval_yml["targets"][target]
    assert (
        set(eval_samples) == parquet_clip_ids
    ), "mismatch found between parquet and yaml"
    validated_clip_ids = set(validation_csv[0])
    assert (
        set(eval_samples) == validated_clip_ids
    ), "mismatch found between yaml and validation csv"

    print(len(eval_samples), "total samples")

    # select clips to remove
    bad_samples = validation_csv[validation_csv[1] == "bad"]
    print(f"{bad_samples.shape[0]} bad samples found")

    cleaned_parquet = eval_parquet[~eval_parquet.clip_id.isin(bad_samples[0])]

    good_samples = validation_csv[validation_csv[1] == "good"][0].tolist()

    if (cleaned_parquet.shape[0] == eval_parquet.shape[0]) and len(good_samples) == len(
        eval_yml["targets"][target]
    ):
        print("no bad samples found, nothing to do")
        return

    # cleaned yaml
    eval_yml["targets"][target] = good_samples

    assert len(good_samples) == cleaned_parquet.shape[0], "cleaned mismatch"

    percent_good = len(good_samples) / len(eval_samples) * 100
    print(f"{cleaned_parquet.shape[0]} samples are valid ({percent_good:0.2f}%)")

    # overwrite eval yaml file
    if not dryrun:
        print("removing", len(bad_samples), "bad samples from", target)
        print("overwriting yaml", eval_yml_file)
        eval_yml_file.write_text(yaml.dump(eval_yml))
        print("overwriting parquet", parquet_file)
        cleaned_parquet.to_parquet(parquet_file)


target_validation_filter("episode")
# %%
for target in ["episode", "job", "restaurant", "fifty", "route"]:
    target_validation_filter(target, dryrun=True)
# %%
raise ValueError("caution - this overwrites")
for target in ["episode", "job", "restaurant", "fifty", "route"]:
    target_validation_filter(target, dryrun=False)
# %%
