# based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/test_streaming_accuracy.py

#%%
from dataclasses import dataclass, asdict
import logging
import sox
import datetime
import os
import multiprocessing
import sys
import shutil
import pprint
import pickle
import glob
import subprocess
import pathlib
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

sns.set()
sns.set_palette("bright")

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
import input_data
from accuracy_utils import StreamingAccuracyStats
from single_target_recognize_commands import (
    SingleTargetRecognizeCommands,
    RecognizeResult,
)

# %%
@dataclass(frozen=True)
class FlagTest:
    wav: os.PathLike
    ground_truth: os.PathLike
    target_keyword: str
    detection_thresholds: List[float]
    clip_duration_ms: int = 1000
    clip_stride_ms: int = 20  # window_stride_ms in model_settings
    # average_window_duration_ms: int = 500
    average_window_duration_ms: int = 100
    suppression_ms: int = 500
    time_tolerance_ms: int = 1500
    minimum_count: int = 4

    def labels(self) -> List[str]:
        return [
            input_data.SILENCE_LABEL,
            input_data.UNKNOWN_WORD_LABEL,
            self.target_keyword,
        ]


def calculate_streaming_accuracy(
    model, model_settings, FLAGS, existing_inferences=None
):
    wav_loader = tf.io.read_file(FLAGS.wav)
    (audio, sample_rate) = tf.audio.decode_wav(wav_loader, desired_channels=1)
    sample_rate = sample_rate.numpy()
    audio = audio.numpy().flatten()

    # Init instance of StreamingAccuracyStats and load ground truth.
    # number of samples in the entire audio file
    data_samples = audio.shape[0]
    # number of samples in one utterance (e.g., a 1 second clip)
    clip_duration_samples = int(FLAGS.clip_duration_ms * sample_rate / 1000)
    # number of samples in one stride
    clip_stride_samples = int(FLAGS.clip_stride_ms * sample_rate / 1000)
    # leave space for one full clip at end
    audio_data_end = data_samples - clip_duration_samples

    if existing_inferences is not None:
        inferences = existing_inferences
    else:
        # num spectrograms: in the range expression below, if there is a remainder
        # in audio_data_end/clip_stride_samples, we need one additional slot
        # for a spectrogram
        spectrograms = np.zeros(
            (
                int(np.ceil(audio_data_end / clip_stride_samples)),
                model_settings["spectrogram_length"],
                model_settings["fingerprint_width"],
            )
        )
        print("building spectrograms", flush=True)
        # Inference along audio stream.
        for ix, audio_data_offset in enumerate(
            range(0, audio_data_end, clip_stride_samples)
        ):
            input_start = audio_data_offset
            input_end = audio_data_offset + clip_duration_samples
            spectrograms[ix] = input_data.to_micro_spectrogram(
                model_settings, audio[input_start:input_end]
            )

        inferences = model.predict(spectrograms[:, :, :, np.newaxis])
        print("inferences complete", flush=True)

    results = {}
    for threshold in FLAGS.detection_thresholds:
        stats = StreamingAccuracyStats(target_keyword=FLAGS.target_keyword)
        stats.read_ground_truth_file(FLAGS.ground_truth)
        recognize_element = RecognizeResult()
        recognize_commands = SingleTargetRecognizeCommands(
            labels=FLAGS.labels(),
            average_window_duration_ms=FLAGS.average_window_duration_ms,
            detection_threshold=threshold,
            suppression_ms=FLAGS.suppression_ms,
            minimum_count=FLAGS.minimum_count,
            target_id=2,
        )
        all_found_words = []
        all_found_words_w_confidences = []
        # calculate statistics using inferences
        for ix, audio_data_offset in enumerate(
            range(0, audio_data_end, clip_stride_samples)
        ):
            output_softmax = inferences[ix]
            current_time_ms = int(audio_data_offset * 1000 / sample_rate)
            recognize_commands.process_latest_result(
                output_softmax, current_time_ms, recognize_element
            )
            if (
                recognize_element.is_new_command
                and recognize_element.found_command != "_silence_"
            ):
                all_found_words.append(
                    [recognize_element.found_command, current_time_ms]
                )
                all_found_words_w_confidences.append(
                    [
                        recognize_element.found_command,
                        current_time_ms,
                        recognize_element.score,
                    ]
                )
                stats.calculate_accuracy_stats(
                    all_found_words, current_time_ms, FLAGS.time_tolerance_ms
                )
                recognition_state = stats.delta()
                # fmt: off
                # print( "{}ms {}:{}{}".format(current_time_ms,recognize_element.founded_command,recognize_element.score,recognition_state,) )
                # fmt: on
                # stats.print_accuracy_stats()
        print("DONE", threshold)
        # calculate final stats for full wav file:
        stats.calculate_accuracy_stats(all_found_words, -1, FLAGS.time_tolerance_ms)
        stats.print_accuracy_stats()
        results[threshold] = (stats, all_found_words, all_found_words_w_confidences)
    return results, inferences


# %%

# %%


@dataclass
class StreamTarget:
    target_lang: str
    target_word: str
    model_path: os.PathLike
    stream_wav: os.PathLike
    stream_label: os.PathLike
    destination_result_pkl: os.PathLike
    destination_result_inferences: os.PathLike


def eval_stream_test(st: StreamTarget):
    tf.get_logger().setLevel(logging.ERROR)
    model = tf.keras.models.load_model(st.model_path)
    tf.get_logger().setLevel(logging.INFO)

    model_settings = input_data.standard_microspeech_model_settings(label_count=3)

    flags = FlagTest(
        wav=str(st.stream_wav),
        ground_truth=str(st.stream_label),
        target_keyword=st.target_word,
        detection_thresholds=np.linspace(0.05, 1, 20).tolist(),  # step threshold 0.05
    )

    if os.path.isfile(st.destination_result_pkl):
        print("results already present", st.destination_result_pkl, flush=True)
        return
    print("SAVING results TO\n", st.destination_result_pkl)
    inferences_exist = False
    if os.path.isfile(st.destination_result_inferences):
        print("inferences already present", flush=True)
        loaded_inferences = np.load(st.destination_result_pkl)
        inferences_exist = True
    else:
        print("SAVING inferences TO\n", st.destination_result_inferences, flush=True)

    results = {}
    if inferences_exist:
        results[st.target_word], _ = calculate_streaming_accuracy(
            model, model_settings, flags, loaded_inferences
        )
    else:
        results[st.target_word], inferences = calculate_streaming_accuracy(
            model, model_settings, flags
        )
    end = datetime.datetime.now()

    with open(st.destination_result_pkl, "wb") as fh:
        pickle.dump(results, fh)
    if not inferences_exist:
        np.save(st.destination_result_inferences, inferences)

    # https://keras.io/api/utils/backend_utils/
    tf.keras.backend.clear_session()

def batch_streaming_analysis():
    batch_data_to_process = []

    # fmt: off
    sse = Path("/home/mark/tinyspeech_harvard/paper_data/streaming_batch_sentences/")
    dest_dir = Path("/home/mark/tinyspeech_harvard/paper_data/results_streaming_batch_sentences/")
    # fmt: on

    for ix, lang_dir in enumerate(os.listdir(sse)):
        target_lang = lang_dir.split("_")[-1]
        for word_dir in os.listdir(sse / lang_dir):
            target_word = word_dir.split("_")[-1]
            print(target_lang, target_word)

            # fmt: off
            model_files = os.listdir(sse / lang_dir / word_dir / "model")
            if len(model_files) != 1:
                raise ValueError("extra models or no models")
            model_path = sse / lang_dir / word_dir / "model" / model_files[0]

            stream_wav =  sse / lang_dir / word_dir / "streaming_test.wav"
            stream_label = sse / lang_dir / word_dir / "streaming_labels.txt"
            assert os.path.isfile(stream_wav) and os.path.isfile(stream_label), "missing stream info"

            destination_result_pkl = dest_dir / lang_dir / word_dir / "stream_results.pkl"
            destination_result_inferences = dest_dir / lang_dir / word_dir / "raw_inferences.npy"
            assert not os.path.isfile(destination_result_pkl) and not os.path.isfile(destination_result_inferences), "result data already present"
            # fmt: on

            d = StreamTarget(
                target_lang=target_lang,
                target_word=target_word,
                model_path=model_path,
                stream_wav=stream_wav,
                stream_label=stream_label,
                destination_result_pkl=destination_result_pkl,
                destination_result_inferences=destination_result_inferences,
            )
            batch_data_to_process.append(d)

    np.random.shuffle(batch_data_to_process)

    n_wavs = len(batch_data_to_process)
    print("n wavs", n_wavs, flush=True)

    batchdata_file = (
        "/home/mark/tinyspeech_harvard/paper_data/data_streaming_batch_sentences.pkl"
    )
    assert not os.path.exists(batchdata_file), f"{batchdata_file} already exists"
    with open(batchdata_file, "wb") as fh:
        pickle.dump(batch_data_to_process, fh)

    for ix, d in enumerate(batch_data_to_process):
        print(
            f"\n\n\n::::::::::::::::: {ix} / {n_wavs} ::::{d.target_lang} - {d.target_word} ::::: ",
            flush=True,
        )
        start = datetime.datetime.now()

        # make result dir
        result_dir = os.path.split(d.destination_result_pkl)[0]
        print("making dir", result_dir, flush=True)
        os.makedirs(result_dir, exist_ok=True)

        p = multiprocessing.Process(target=eval_stream_test, args=(d,))
        p.start()
        p.join()

        end = datetime.datetime.now()
        print("time elapsed", end - start)

# %%

# %%
# python embedding/batch_streaming_analysis.py > /home/mark/tinyspeech_harvard/paper_data/results_multilang_sentence.log
if __name__ == "__main__":
    batch_streaming_analysis()
