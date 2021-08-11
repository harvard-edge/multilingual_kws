#%%
from dataclasses import dataclass
import logging
import datetime
import os
import multiprocessing
import pickle
import glob
import subprocess
from pathlib import Path
from typing import List
import sys

import numpy as np
import tensorflow as tf

import input_data
import transfer_learning
import batch_streaming_analysis as sa

# %%


@dataclass(frozen=True)
class TLData:
    train_files: List[os.PathLike]
    val_files: List[os.PathLike]
    n_batches: int
    n_epochs: int
    model_dest_dir: os.PathLike
    primary_lr: float
    backprop_into_embedding: bool
    embedding_lr: float
    with_context: bool
    target: str
    stream_targets: List[sa.StreamTarget]
    batch_size: int = 64


def train_process(d: TLData):
    print("\n\n----starting \n", d.train_files[0], "\n\n")

    # check if this has been processed already
    for t in d.stream_targets:
        if os.path.isfile(t.destination_result_pkl):
            print("results already present", t.destination_result_pkl, flush=True)
            return

    traindir = Path(f"/home/mark/tinyspeech_harvard/multilingual_embedding_wc")

    unknown_files = []
    unknown_files_dir = Path("/home/mark/tinyspeech_harvard/unknown_files/")
    with open(unknown_files_dir / "unknown_files.txt", "r") as fh:
        for w in fh.read().splitlines():
            unknown_files.append(str(unknown_files_dir / w))

    base_model_path = traindir / "models" / "multilingual_context_73_0.8011"
    model_settings = input_data.standard_microspeech_model_settings(3)

    name, model, details = transfer_learning.transfer_learn(
        target=d.target,
        train_files=d.train_files,
        val_files=d.val_files,
        unknown_files=unknown_files,
        num_epochs=d.n_epochs,
        num_batches=d.n_batches,
        batch_size=d.batch_size,
        primary_lr=d.primary_lr,
        backprop_into_embedding=d.backprop_into_embedding,
        embedding_lr=d.embedding_lr,
        model_settings=model_settings,
        base_model_path=base_model_path,
        base_model_output="dense_2",
        csvlog_dest=d.model_dest_dir / "log.csv",
    )
    # skip saving model - slow
    # print("saving", name)
    # modelpath = d.model_dest_dir / name
    # model.save(modelpath)

    for t in d.stream_targets:
        sa.eval_stream_test(t, live_model=model)


# %%

def generate_and_run():
    # fmt: off
    ine_sentences = Path("/home/mark/tinyspeech_harvard/paper_data/streaming_batch_sentences/")
    ooe_sentences = Path("/home/mark/tinyspeech_harvard/paper_data/ooe_streaming_batch_sentences/")
    ine_perword   = Path("/home/mark/tinyspeech_harvard/paper_data/streaming_batch_perword/")
    ooe_perword   = Path("/home/mark/tinyspeech_harvard/paper_data/ooe_streaming_batch_perword/")
    # fmt: on

    # sources = zip(
    #     ["ine_sentences", "ooe_sentences", "ine_perword", "ooe_perword"],
    #     [ine_sentences, ooe_sentences, ine_perword, ooe_perword],
    # )

    models_dest = Path("/home/mark/tinyspeech_harvard/paper_data/context_models")

    train_targets = []
    # n_shots training files only exist for {ine,ooe}_sentences, the same model is reused for _perword
    for model_source_type, sse in zip(
        ["ine_sentences", "ooe_sentences"], [ine_sentences, ooe_sentences]
    ):
        for ix, lang_dir in enumerate(os.listdir(sse)):
            if not os.path.isdir(sse / lang_dir):
                continue  # skip the data generator shellscript and the logfiles
            target_lang = lang_dir.split("_")[-1]
            for word_dir in os.listdir(sse / lang_dir):
                target_word = word_dir.split("_")[-1]
                print(target_lang, target_word)

                model_dir = models_dest / target_lang / target_word
                os.makedirs(model_dir, exist_ok=True)
                if len(os.listdir(model_dir)) != 0:
                    raise ValueError("models already present")

                trainfile_paths = sse / lang_dir / word_dir / "n_shots"
                val_paths = sse / lang_dir / word_dir / "val"
                train_files = [
                    str(trainfile_paths / f) for f in os.listdir(trainfile_paths)
                ]
                assert len(train_files) == 5, "missing train files"
                if len(os.listdir(val_paths)) == 0:
                    val_files = train_files
                else:
                    val_files = [str(val_paths / f) for f in os.listdir(val_paths)]

                # collect streaming targets
                if model_source_type == "ine_sentences":
                    stream_sources = zip(
                        ["ine_sentences", "ine_perword"], [ine_sentences, ine_perword]
                    )
                else:
                    stream_sources = zip(
                        ["ooe_sentences", "ooe_perword"], [ooe_sentences, ooe_perword]
                    )

                stream_targets = []
                for stream_source_name, source in stream_sources:
                    streamwav = source / lang_dir / word_dir / "streaming_test.wav"
                    gt = source / lang_dir / word_dir / "streaming_labels.txt"
                    assert os.path.isfile(streamwav), f"{streamwav} not present"
                    assert os.path.isfile(gt)

                    # fmt: off
                    dp = models_dest / target_lang / target_word / (stream_source_name + "_results" + ".pkl")
                    di = models_dest / target_lang / target_word / (stream_source_name + "_inferences" + ".npy")
                    # fmt: on
                    assert not os.path.isfile(dp)
                    assert not os.path.isfile(di)
                    flags = sa.StreamFlags(
                        wav=str(streamwav),
                        ground_truth=str(gt),
                        target_keyword=target_word,
                        detection_thresholds=np.linspace(
                            0.05, 1, 20
                        ).tolist(),  # step threshold 0.05
                        # average_window_duration_ms = 500
                        average_window_duration_ms=100,
                        suppression_ms=500,
                        time_tolerance_ms=750,  # only used when graphing
                    )
                    streamtarget = sa.StreamTarget(
                        target_lang=target_lang,
                        target_word=target_word,
                        model_path=None,
                        destination_result_pkl=dp,
                        destination_result_inferences=di,
                        stream_flags=[flags],
                    )
                    stream_targets.append(streamtarget)

                d = TLData(
                    train_files=train_files,
                    val_files=val_files,
                    n_batches=1,
                    n_epochs=4,
                    model_dest_dir=model_dir,
                    primary_lr=0.001,
                    backprop_into_embedding=False,
                    embedding_lr=0,
                    with_context=True,
                    target=target_word,
                    stream_targets=stream_targets,
                )
                train_targets.append(d)

    np.random.seed(50)
    np.random.shuffle(train_targets)
    batchdata_file = "/home/mark/tinyspeech_harvard/paper_data/context_batchdata.pkl"
    assert not os.path.exists(batchdata_file), f"{batchdata_file} already exists"
    with open(batchdata_file, "wb") as fh:
        pickle.dump(train_targets, fh)

    total = len(train_targets)
    for ix, d in enumerate(train_targets):
        start = datetime.datetime.now()
        p = multiprocessing.Process(target=train_process, args=(d,))
        p.start()
        p.join()
        end = datetime.datetime.now()
        print(f"\n\n::::::: {ix} / {total} elapsed time", end - start, "\n\n")


def resume_run():
    batchdata_file = "/home/mark/tinyspeech_harvard/paper_data/context_batchdata.pkl"
    with open(batchdata_file, "rb") as fh:
        train_targets = pickle.load(fh)

    total = len(train_targets)
    for ix, d in enumerate(train_targets):
        start = datetime.datetime.now()
        p = multiprocessing.Process(target=train_process, args=(d,))
        p.start()
        p.join()
        end = datetime.datetime.now()
        print(f"\n\n::::::: {ix} / {total} elapsed time", end - start, "\n\n")

if __name__ == "__main__":
    resume_run()