# %%
from pathlib import Path
import pandas as pd
import yaml
import subprocess


# %%
# normalization
for lang in ["id", "pt"]:
    for split in ["validation", "test"]:
        wavs = (
            Path.home()
            / f"h/dataperf/target_listening_data/unnormalized/{lang}/{split}"
        )
        wavs = list(wavs.glob("*.wav"))
        print(lang, split, len(wavs))
        for wav in wavs:
            dest = (
                Path.home()
                / f"h/dataperf/target_listening_data/normalized/{lang}/{split}/{wav.name}"
            )
            # http://johnriselvato.com/ffmpeg-how-to-normalize-audio/
            subprocess.run(
                [
                    "/Users/mark/miniconda3/envs/ffmpeg/bin/ffmpeg",
                    "-i",
                    str(wav),
                    "-af",
                    "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-c:a",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    str(dest),
                ]
            )
print("Done")


# %%


def target_validation_filter(
    target, listening_results_basedir, experiment_basedir, dryrun=True
):
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
        # not strictly necessary to overwrite the parquet
        print("overwriting parquet", parquet_file)
        cleaned_parquet.to_parquet(parquet_file)


# target_validation_filter("episode")
# %%
# english
raise ValueError("caution - this overwrites")
listening_results_basedir = (
    Path.home() / "tinyspeech_harvard/dataperf/listening_results/round1/"
)

experiment_basedir = (
    Path.home() / "tinyspeech_harvard/dataperf/preliminary_evaluation_dataset"
)
list(listening_results_basedir.iterdir())
for target in ["episode", "job", "restaurant", "fifty", "route"]:
    target_validation_filter(target, dryrun=False)
# %%
# indonesian
raise ValueError("caution - this overwrites")
listening_results_basedir = Path.home() / "h/dataperf/id_pt_listening_results/"
experiment_basedir = Path.home() / "h/dataperf/dataperf_id_data"
list(listening_results_basedir.iterdir())
for target in ["karena", "sangat", "bahasa", "belajar", "kemarin"]:
    target_validation_filter(
        target,
        listening_results_basedir=listening_results_basedir,
        experiment_basedir=experiment_basedir,
        dryrun=False,
    )

# %%
# portuguese
raise ValueError("caution - this overwrites")
listening_results_basedir = Path.home() / "h/dataperf/id_pt_listening_results/"
experiment_basedir = Path.home() / "h/dataperf/dataperf_pt_data"
list(listening_results_basedir.iterdir())
for target in ["pessoas", "grupo", "camisa", "tempo", "andando"]:
    target_validation_filter(
        target,
        listening_results_basedir=listening_results_basedir,
        experiment_basedir=experiment_basedir,
        dryrun=False,
    )

# %%
