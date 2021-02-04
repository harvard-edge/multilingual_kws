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
        stats = StreamingAccuracyStats(target_keyword=FLAGS.target_keyword)
        stats.read_ground_truth_file(FLAGS.ground_truth)
        recognize_element = RecognizeResult()
        recognize_commands = RecognizeCommands(
            labels=FLAGS.labels(),
            average_window_duration_ms=FLAGS.average_window_duration_ms,
            detection_threshold=threshold,
            suppression_ms=FLAGS.suppression_ms,
            minimum_count=4,
        )
        all_found_words = []
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
                # stats.print_accuracy_stats()
        print("DONE", threshold)
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

#%%  print results
# msg, statdict = stats.print_accuracy_stats().values()[0]
# print("matched", statdict["matched"])
# print("wrong", statdict["wrong"])
# print(statdict)


############################################################################

#%%  long pauses between also
modelname = "xfer_5_shot_6_epochs_also_val_acc_0.89"
model_path = f"/home/mark/tinyspeech_harvard/xfer_efnet_5/models/{modelname}"

tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(model_path)
tf.get_logger().setLevel(logging.INFO)

flags = FlagTest(
    wav=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/also2/streaming_test.wav",
    ground_truth=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/also2/streaming_labels.txt",
    target_keyword="also",
    # detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    detection_thresholds=[0.6],
)
results = calculate_streaming_accuracy(model, audio_dataset, FLAGS=flags)


#%%

#%%
target_match_over_gt_positives = []
false_positives_over_silence_or_unknown = []
thresh_labels = []
target = "also"

fig = go.Figure()
for threshold, stats in results.items():
    msg, statdict = stats.print_accuracy_stats()
    n_targets = statdict["num_groundtruth_target"]
    n_silence_unknown = statdict["num_groundtruth_unknown_or_silence"]
    n_correct = statdict["matched"][target]
    n_false_positives = (
        statdict["wrong"][input_data.SILENCE_LABEL]
        + statdict["wrong"][input_data.UNKNOWN_WORD_LABEL]
    )
    target_match_over_gt_positives.append(n_correct / n_targets)
    false_positives_over_silence_or_unknown.append(
        n_false_positives / n_silence_unknown
    )
    thresh_labels.append(f"thresh: {threshold}")
fig.add_trace(
    go.Scatter(
        x=false_positives_over_silence_or_unknown,
        y=target_match_over_gt_positives,
        text=thresh_labels,
        name=target,
    )
)
fig.update_layout(
    xaxis_title="(false positive target matches)/(# GT silence, unknown)",
    yaxis_title="(target matches)/(# GT targets)",
)
fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[0, 1])
fig

##############################################################################

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
        detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
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
        false_positives_over_silence_or_unknown.append(
            n_false_positives / n_silence_unknown
        )
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
fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[0, 1])
fig

#%%
fig.write_html(
    "/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming_performance_aggregate.html"
)


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


######################################################################


#%% internals of calculating stats
modelname = "xfer_5_shot_6_epochs_also_val_acc_0.89"
model_path = f"/home/mark/tinyspeech_harvard/xfer_efnet_5/models/{modelname}"

tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(model_path)
tf.get_logger().setLevel(logging.INFO)

FLAGS = FlagTest(
    wav=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/also2/streaming_test.wav",
    ground_truth=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/also2/streaming_labels.txt",
    target_keyword="also",
    # detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    detection_thresholds=[0.6],
)
wav_loader = tf.io.read_file(FLAGS.wav)
(audio, sample_rate) = tf.audio.decode_wav(wav_loader, desired_channels=1)
sample_rate = sample_rate.numpy()
audio = audio.numpy().flatten()

# Init instance of StreamingAccuracyStats and load ground truth.
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

#prs = [[24480, np.array([0.97381717, 0.01749072, 0.00869214], dtype=np.float32)], [24500, np.array([0.9653848 , 0.0225329 , 0.01208234], dtype=np.float32)], [24520, np.array([0.9693769 , 0.02020488, 0.01041819], dtype=np.float32)], [24540, np.array([0.90422493, 0.05676545, 0.03900962], dtype=np.float32)], [24560, np.array([0.8826283 , 0.06856677, 0.04880488], dtype=np.float32)], [24580, np.array([0.9340749 , 0.04020728, 0.02571781], dtype=np.float32)], [24600, np.array([0.9071234 , 0.05524103, 0.03763561], dtype=np.float32)], [24620, np.array([0.7386729 , 0.14834712, 0.11297993], dtype=np.float32)], [24640, np.array([0.70328534, 0.15754056, 0.13917409], dtype=np.float32)], [24660, np.array([0.47098544, 0.2969344 , 0.23208016], dtype=np.float32)], [24680, np.array([0.2131788 , 0.39437377, 0.39244738], dtype=np.float32)], [24700, np.array([0.15790263, 0.5700083 , 0.27208915], dtype=np.float32)], [24720, np.array([0.11816145, 0.66927344, 0.21256515], dtype=np.float32)], [24740, np.array([0.13384889, 0.6829233 , 0.18322779], dtype=np.float32)], [24760, np.array([0.11229166, 0.7446161 , 0.14309229], dtype=np.float32)], [24780, np.array([0.10971147, 0.7581488 , 0.13213971], dtype=np.float32)], [24800, np.array([0.11647787, 0.68431634, 0.19920574], dtype=np.float32)], [24820, np.array([0.11367017, 0.73021716, 0.15611266], dtype=np.float32)], [24840, np.array([0.11607315, 0.6881232 , 0.1958036 ], dtype=np.float32)], [24860, np.array([0.12394444, 0.6511759 , 0.2248797 ], dtype=np.float32)], [24880, np.array([0.10275374, 0.6805507 , 0.21669555], dtype=np.float32)], [24900, np.array([0.10832468, 0.6424134 , 0.24926198], dtype=np.float32)], [24920, np.array([0.08836265, 0.6254203 , 0.2862171 ], dtype=np.float32)], [24940, np.array([0.12270603, 0.4178935 , 0.45940053], dtype=np.float32)], [24960, np.array([0.11143086, 0.42042932, 0.4681398 ], dtype=np.float32)], [24980, np.array([0.12279104, 0.18111731, 0.69609165], dtype=np.float32)]]
#
#scores = np.array([p[1] for p in prs])
#scores
#%%

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
import accuracy_utils as mau

#%%
import importlib
importlib.reload(mau)


#%%



threshold = 0.6
stats = mau.StreamingAccuracyStats(target_keyword=FLAGS.target_keyword)
stats.read_ground_truth_file(FLAGS.ground_truth)
recognize_element = RecognizeResult()
recognize_commands = RecognizeCommands(
    labels=FLAGS.labels(),
    average_window_duration_ms=FLAGS.average_window_duration_ms,
    detection_threshold=threshold,
    suppression_ms=FLAGS.suppression_ms,
    minimum_count=4,
)
all_found_words = []
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
        #print(output_softmax, recognize_element.founded_command, recognize_element.score)
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
print("DONE", threshold)
# calculate final stats for full wav file:
stats.calculate_accuracy_stats(all_found_words, -1, FLAGS.time_tolerance_ms)
msg, info = stats.print_accuracy_stats()
info

#%%
print(len(stats._gt_occurrence))
print(len(all_found_words))
num_unknowns = len([w for w in all_found_words if w[0]=="_unknown_"])
print(num_unknowns, num_unknowns/len(all_found_words))

#%%
print({'correct_match_percentage': 88.7218045112782,
 'wrong_match_percentage': 7.518796992481203,
 'howmanyfp': 0,
 'howmanyfn': 7,
 'wrong': {'_silence_': 0, '_unknown_': 2, 'also': 0},
 'matched': {'_silence_': 0, '_unknown_': 56, 'also': 62},
 'num_groundtruth_target': 71,
 'num_groundtruth_unknown_or_silence': 62})

#%%
afw =  [['_unknown_', 2820], ['also', 16580], ['_unknown_', 24980], ['also', 25500]] 
stats._gt_occurrence[:5]

#%%
len(stats._gt_occurrence)
#%%
gt_u, gt_t = [], []
for l, t in stats._gt_occurrence:
    if l=="_unknown_":
        gt_u.append([t, 1])
        gt_t.append([t, 0])
    else:
        gt_u.append([t, 0])
        gt_t.append([t, 1])
gt_u = np.array(gt_u)
gt_t = np.array(gt_t)

#%%
f_u, f_t = [], []
for l, t in all_found_words:
    if l=="_unknown_":
        f_u.append([t, 0.5])
        f_t.append([t, 0])
    else:
        f_u.append([t, 0])
        f_t.append([t, 0.5])
f_u = np.array(f_u)
f_t = np.array(f_t)

#%%
fig,ax = plt.subplots()
ax.plot(gt_u[:,0], gt_u[:,1], label="groundtruth unknown", color="blue")
ax.fill_between(gt_u[:,0], gt_u[:,1], color=(0,0,0.8,0.3))
ax.plot(gt_t[:,0], gt_t[:,1], label="groundtruth target", color="green")
ax.fill_between(gt_t[:,0], gt_t[:,1], color=(0.2,0.8,0,0.3))

ax.plot(f_u[:,0], f_u[:,1], label="found unknown", color="orange")
ax.fill_between(f_u[:,0], f_u[:,1], color="orange")
ax.plot(f_t[:,0], f_t[:,1], label="found target", color="purple")
ax.fill_between(f_t[:,0], f_t[:,1], color="purple")
ax.legend(loc="upper right")
fig.set_size_inches(30,5)
