# based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/test_streaming_accuracy.py


#%%
import argparse
from dataclasses import dataclass
import logging
import os
import sys
import pprint
import pickle
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow._api.v2 import audio

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

sns.set()
sns.set_palette("bright")

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
import input_data
from accuracy_utils import StreamingAccuracyStats

sys.path.insert(
    0, "/home/mark/tinyspeech_harvard/tensorflow/tensorflow/examples/speech_commands/"
)
from recognize_commands import RecognizeCommands, RecognizeResult


#%%
@dataclass(frozen=True)
class FlagTest:
    wav: os.PathLike
    ground_truth: os.PathLike
    target_keyword: str
    detection_thresholds: List[float]
    clip_duration_ms: int = 1000
    clip_stride_ms: int = 20  # window_stride_ms in model_settings
    average_window_duration_ms: int = 500
    suppression_ms: int = 500
    time_tolerance_ms: int = 1500

    def labels(self) -> List[str]:
        return [
            input_data.SILENCE_LABEL,
            input_data.UNKNOWN_WORD_LABEL,
            self.target_keyword,
        ]


def calculate_streaming_accuracy(model, audio_dataset, FLAGS):
    wav_loader = tf.io.read_file(FLAGS.wav)
    (audio, sample_rate) = tf.audio.decode_wav(wav_loader, desired_channels=1)
    sample_rate = sample_rate.numpy()
    audio = audio.numpy().flatten()

    # Init instance of StreamingAccuracyStats and load ground truth.
    stats = StreamingAccuracyStats(target_keyword=FLAGS.target_keyword)
    stats.read_ground_truth_file(FLAGS.ground_truth)
    recognize_element = RecognizeResult()
    all_found_words = []
    data_samples = audio.shape[0]
    clip_duration_samples = int(FLAGS.clip_duration_ms * sample_rate / 1000)
    clip_stride_samples = int(FLAGS.clip_stride_ms * sample_rate / 1000)
    audio_data_end = data_samples - clip_duration_samples

    spectrograms = np.zeros(
        (
            audio_data_end // clip_stride_samples,
            model_settings["spectrogram_length"],
            model_settings["fingerprint_width"],
        )
    )
    print("building spectrograms")
    # Inference along audio stream.
    for ix, audio_data_offset in enumerate(
        range(0, audio_data_end, clip_stride_samples)
    ):
        input_start = audio_data_offset
        input_end = audio_data_offset + clip_duration_samples
        spectrograms[ix] = audio_dataset.to_micro_spectrogram(
            audio[input_start:input_end]
        )

    inferences = model.predict(spectrograms[:, :, :, np.newaxis])
    print("inferences complete")

    results = {}
    for threshold in FLAGS.detection_thresholds:
        recognize_commands = RecognizeCommands(
            labels=FLAGS.labels(),
            average_window_duration_ms=FLAGS.average_window_duration_ms,
            detection_threshold=threshold,
            suppression_ms=FLAGS.suppression_ms,
            minimum_count=4,
        )
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
                and recognize_element.founded_command != "_silence_"
            ):
                all_found_words.append(
                    [recognize_element.founded_command, current_time_ms]
                )
                stats.calculate_accuracy_stats(
                    all_found_words, current_time_ms, FLAGS.time_tolerance_ms
                )
                recognition_state = stats.delta()
                # print(
                #     "{}ms {}:{}{}".format(
                #         current_time_ms,
                #         recognize_element.founded_command,
                #         recognize_element.score,
                #         recognition_state,
                #     )
                # )
                stats.print_accuracy_stats()
        print("DONE")
        # calculate final stats for full wav file:
        stats.calculate_accuracy_stats(all_found_words, -1, FLAGS.time_tolerance_ms)
        stats.print_accuracy_stats()
        results[threshold] = stats
    return results


#%%  MODEL

# xfer_5_shot_6_epochs_also_val_acc_0.89/
# xfer_5_shot_6_epochs_always_val_acc_0.98/
# xfer_5_shot_6_epochs_area_val_acc_0.91/
# xfer_5_shot_6_epochs_between_val_acc_0.91/
# xfer_5_shot_6_epochs_between_val_acc_0.94/
# xfer_5_shot_6_epochs_last_val_acc_0.97/
# xfer_5_shot_6_epochs_long_val_acc_0.95/
# xfer_5_shot_6_epochs_thing_val_acc_0.88/
# xfer_5_shot_6_epochs_will_val_acc_0.87/
m = "xfer_5_shot_6_epochs_always_val_acc_0.98"
model_path = f"/home/mark/tinyspeech_harvard/xfer_efnet_5/models/{m}"

tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(model_path)
tf.get_logger().setLevel(logging.INFO)

#%%  AUDIO_DATASET
model_settings = input_data.prepare_model_settings(
    label_count=100,
    sample_rate=16000,
    clip_duration_ms=1000,
    window_size_ms=30,
    window_stride_ms=20,
    feature_bin_count=40,
    preprocess="micro",
)
bg_datadir = "/home/mark/tinyspeech_harvard/frequent_words/en/clips/_background_noise_/"
audio_dataset = input_data.AudioDataset(
    model_settings,
    ["NONE"],
    bg_datadir,
    unknown_files=[],
    unknown_percentage=0,
    spec_aug_params=input_data.SpecAugParams(percentage=0),
)

#%%  calculate results

flags = FlagTest(
    wav="/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/always/streaming_test.wav",
    ground_truth="/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/always/streaming_labels.txt",
    target_keyword="always",
    detection_thresholds=[0.6],
)

stats = calculate_streaming_accuracy(model, audio_dataset, FLAGS=flags)

#%%  print results
msg, statdict = stats.print_accuracy_stats().values()[0]
print("matched", statdict["matched"])
print("wrong", statdict["wrong"])
print(statdict)


############################################################################

#%%
with open(DESTINATION, "rb") as fh:
    allresults = pickle.load(fh)

#%% COLLECT AGGREGATE STATS

DESTINATION = "/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming_stats_fast.pkl"

models = [
    ("also", "also", "xfer_5_shot_6_epochs_also_val_acc_0.89/"),
    ("always", "always", "xfer_5_shot_6_epochs_always_val_acc_0.98/"),
    ("area", "area", "xfer_5_shot_6_epochs_area_val_acc_0.91/"),
    ("between_1", "between", "xfer_5_shot_6_epochs_between_val_acc_0.91/"),
    ("between_2", "between", "xfer_5_shot_6_epochs_between_val_acc_0.94/"),
    ("last", "last", "xfer_5_shot_6_epochs_last_val_acc_0.97/"),
    ("long", "long", "xfer_5_shot_6_epochs_long_val_acc_0.95/"),
    ("thing", "thing", "xfer_5_shot_6_epochs_thing_val_acc_0.88/"),
    ("will", "will", "xfer_5_shot_6_epochs_will_val_acc_0.87/"),
]

full_results = {}
for resultname, target_keyword, modelname in models:
    model_path = f"/home/mark/tinyspeech_harvard/xfer_efnet_5/models/{modelname}"

    tf.get_logger().setLevel(logging.ERROR)
    model = tf.keras.models.load_model(model_path)
    tf.get_logger().setLevel(logging.INFO)

    full_results[resultname] = {}

    flags = FlagTest(
        wav=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/{target_keyword}/streaming_test.wav",
        ground_truth=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/{target_keyword}/streaming_labels.txt",
        target_keyword=target_keyword,
        detection_thresholds=np.linspace(0, 1, 21).aslist(),  # step threshold 0.05
    )

    print("-:::::::::::::-", resultname)
    results = calculate_streaming_accuracy(model, audio_dataset, FLAGS=flags)

    full_results[resultname] = results

with open(DESTINATION, "wb") as fh:
    pickle.dump(full_results, fh)
print("FULL RESULTS DONE")

#%%
for resultname, results_per_threshold in full_results.items():
    print(resultname)
    target_match_over_gt_positives = []
    false_positives_over_silence_or_unknown = []
    thresh_labels = []
    for (thresh, stats) in results_per_threshold.items():
        target = stats[0]
        statdict = stats[2]
        n_targets = statdict["num_groundtruth_target"]
        n_silence_unknown = statdict["num_groundtruth_unknown_or_silence"]
        n_correct = statdict["matched"][target]
        n_false_positives = (
            statdict["wrong"][input_data.SILENCE_LABEL]
            + statdict["wrong"][input_data.UNKNOWN_WORD_LABEL]
        )
        print(target, n_targets, n_silence_unknown, n_correct, n_false_positives)
        break
    break


#%%
fig = go.Figure()
for resultname, results_per_threshold in full_results.items():
    print(resultname)
    target_match_over_gt_positives = []
    false_positives_over_silence_or_unknown = []
    thresh_labels = []
    for (thresh, stats) in results_per_threshold.items():
        target = stats[0]
        statdict = stats[2]
        n_targets = statdict["num_groundtruth_target"]
        n_silence_unknown = statdict["num_groundtruth_unknown_or_silence"]
        n_correct = statdict["matched"][target]
        n_false_positives = (
            statdict["wrong"][input_data.SILENCE_LABEL]
            + statdict["wrong"][input_data.UNKNOWN_WORD_LABEL]
        )
        target_match_over_gt_positives.append(n_correct / n_targets)
        false_positives_over_silence_or_unknown.append(n_false_positives / n_silence_unknown)
        thresh_labels.append(f"thresh: {thresh}")
    fig.add_trace(
        go.Scatter(
            x=false_positives_over_silence_or_unknown,
            y=target_match_over_gt_positives,
            text=thresh_labels,
            name=resultname,
        )
    )
fig.update_layout(
    xaxis_title="(false positive target matches)/(# GT silence, unknown)",
    yaxis_title="(target matches)/(# GT targets)",
)
fig.update_xaxes(range=[0,1])
fig.update_yaxes(range=[0,1])
fig

#%%
fig.write_html("/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming_performance_aggregate.html")


#%%
with np.printoptions(precision=3, suppress=True):
    pprint.pprint(full_results)
#%%
def markstats(FLAGS, stats):
    cliplength_s = 600.0  # seconds
    false_accepts_per_second = stats._how_many_fp / cliplength_s
    false_accepts_per_hour = false_accepts_per_second * 3600.0
    false_rejects_per_instance = stats._how_many_fn / stats._how_many_gt
    correct_match_percentage = stats._how_many_c / stats._how_many_gt
    #   with open("./markstats/stats.csv", 'a') as fh:
    #     print(f"\n\n\n  {FLAGS.detection_threshold},{false_accepts_per_hour},{false_rejects_per_instance},{correct_match_percentage}  \n\n\n")
    #     fh.write(f"{FLAGS.detection_threshold},{false_accepts_per_hour},{false_rejects_per_instance},{correct_match_percentage}\n")
    report = stats.print_accuracy_stats()


#   with open("./markstats/report.txt", 'a') as fh:
#     fh.write("\n" + str(FLAGS.detection_threshold) + "\n" + report + "\n")
#     break

