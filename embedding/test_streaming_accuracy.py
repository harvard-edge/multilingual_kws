# based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/test_streaming_accuracy.py


#%%
import argparse
from dataclasses import dataclass
import logging
import os
import sys
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

sys.path.insert(
    0, "/home/mark/tinyspeech_harvard/tensorflow/tensorflow/examples/speech_commands/"
)
from recognize_commands import RecognizeCommands, RecognizeResult


#%%

tf.config.list_physical_devices("GPU")

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


def calculate_streaming_accuracy(model, model_settings, FLAGS):
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
    print("building spectrograms")
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
                and recognize_element.founded_command != "_silence_"
            ):
                all_found_words.append(
                    [recognize_element.founded_command, current_time_ms]
                )
                all_found_words_w_confidences.append(
                    [recognize_element.founded_command, current_time_ms, recognize_element.score]
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
        results[threshold] = (stats, all_found_words, all_found_words_w_confidences)
    return results, inferences


# def generate_inferences_offline(model, model_settings, FLAGS):
#     wav_loader = tf.io.read_file(FLAGS.wav)
#     (audio, sample_rate) = tf.audio.decode_wav(wav_loader, desired_channels=1)
#     sample_rate = sample_rate.numpy()
#     audio = audio.numpy().flatten()

#     # Init instance of StreamingAccuracyStats and load ground truth.
#     # number of samples in the entire audio file
#     data_samples = audio.shape[0]
#     print("data samples", data_samples)
#     # number of samples in one utterance (e.g., a 1 second clip)
#     clip_duration_samples = int(FLAGS.clip_duration_ms * sample_rate / 1000)
#     # number of samples in one stride
#     clip_stride_samples = int(FLAGS.clip_stride_ms * sample_rate / 1000)
#     # leave space for one full clip at end
#     audio_data_end = data_samples - clip_duration_samples

#     # num spectrograms: in the range expression below, if there is a remainder
#     # in audio_data_end/clip_stride_samples, we need one additional slot
#     # for a spectrogram
#     spectrograms = np.zeros(
#         (
#             int(np.ceil(audio_data_end / clip_stride_samples)),
#             model_settings["spectrogram_length"],
#             model_settings["fingerprint_width"],
#         )
#     )
#     # print("building spectrograms")
#     # # Inference along audio stream.
#     # for ix, audio_data_offset in enumerate(
#     #     range(0, audio_data_end, clip_stride_samples)
#     # ):
#     #     input_start = audio_data_offset
#     #     input_end = audio_data_offset + clip_duration_samples
#     #     spectrograms[ix] = input_data.to_micro_spectrogram(
#     #         model_settings, audio[input_start:input_end]
#     #     )
#     # inferences = model.predict(spectrograms[:, :, :, np.newaxis])
#     # print("inferences complete")
#     return spectrograms

# def calculate_inferences_offline(inferences, FLAGS):
#     wav_loader = tf.io.read_file(FLAGS.wav)
#     (audio, sample_rate) = tf.audio.decode_wav(wav_loader, desired_channels=1)
#     sample_rate = sample_rate.numpy()
#     audio = audio.numpy().flatten()

#     # Init instance of StreamingAccuracyStats and load ground truth.
#     # number of samples in the entire audio file
#     data_samples = audio.shape[0]
#     # number of samples in one utterance (e.g., a 1 second clip)
#     clip_duration_samples = int(FLAGS.clip_duration_ms * sample_rate / 1000)
#     # number of samples in one stride
#     clip_stride_samples = int(FLAGS.clip_stride_ms * sample_rate / 1000)
#     # leave space for one full clip at end
#     audio_data_end = data_samples - clip_duration_samples

#     results = {}
#     for threshold in FLAGS.detection_thresholds:
#         stats = StreamingAccuracyStats(target_keyword=FLAGS.target_keyword)
#         stats.read_ground_truth_file(FLAGS.ground_truth)
#         recognize_element = RecognizeResult()
#         recognize_commands = RecognizeCommands(
#             labels=FLAGS.labels(),
#             average_window_duration_ms=FLAGS.average_window_duration_ms,
#             detection_threshold=threshold,
#             suppression_ms=FLAGS.suppression_ms,
#             minimum_count=4,
#         )
#         all_found_words = []
#         # calculate statistics using inferences
#         for ix, audio_data_offset in enumerate(
#             range(0, audio_data_end, clip_stride_samples)
#         ):
#             output_softmax = inferences[ix]
#             current_time_ms = int(audio_data_offset * 1000 / sample_rate)
#             recognize_commands.process_latest_result(
#                 output_softmax, current_time_ms, recognize_element
#             )
#             if (
#                 recognize_element.is_new_command
#                 and recognize_element.founded_command != "_silence_"
#             ):
#                 all_found_words.append(
#                     [recognize_element.founded_command, current_time_ms]
#                 )
#                 stats.calculate_accuracy_stats(
#                     all_found_words, current_time_ms, FLAGS.time_tolerance_ms
#                 )
#                 recognition_state = stats.delta()
#                 # print(
#                 #     "{}ms {}:{}{}".format(
#                 #         current_time_ms,
#                 #         recognize_element.founded_command,
#                 #         recognize_element.score,
#                 #         recognition_state,
#                 #     )
#                 # )
#                 # stats.print_accuracy_stats()
#         print("DONE", threshold)
#         # calculate final stats for full wav file:
#         stats.calculate_accuracy_stats(all_found_words, -1, FLAGS.time_tolerance_ms)
#         stats.print_accuracy_stats()
#         results[threshold] = (stats, all_found_words)
#     return results


#%%
model_settings = input_data.standard_microspeech_model_settings(label_count=3)

#%%
############################################################################
#    full sentence streaming test
############################################################################

target = "rebecca"

sse = pathlib.Path("/home/mark/tinyspeech_harvard/streaming_sentence_experiments/")
base_dir = sse / target
model_dir = sse / target / "model"
model_name = os.listdir(model_dir)[0]
print(model_name)
model_path = model_dir / model_name
print(model_path)

DESTINATION = base_dir / "stream_results.pkl"
print("SAVING TO", DESTINATION)

#%%
tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(model_path)
tf.get_logger().setLevel(logging.INFO)

# t_min = 0.4
# t_max = 0.95
# t_steps = int(np.ceil((t_max - t_min) / 0.05)) + 1
# threshs = np.linspace(t_min, t_max, t_steps).tolist()
flags = FlagTest(
    wav=str(base_dir / "stream.wav"),
    ground_truth=str(base_dir / "labels.txt"),
    target_keyword=target,
    # detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    #detection_thresholds=threshs,  # step threshold 0.05
    detection_thresholds=[0.65],  # step threshold 0.05
)
results = {}
results[target], inferences = calculate_streaming_accuracy(model, model_settings, flags)

with open(DESTINATION, "wb") as fh:
    pickle.dump(results, fh)
np.save(base_dir / "raw_inferences.npy", inferences)

#%%
found_words_w_confidences = results[target][0.65][2]
with open(base_dir / f"found_words_w_confidences_{target}.py", "wb") as fh:
    pickle.dump(found_words_w_confidences, fh)


#%%
target = "merchant"

sse = pathlib.Path("/home/mark/tinyspeech_harvard/streaming_sentence_experiments/")
base_dir = sse / target
model_dir = sse / target / "model"
model_name = os.listdir(model_dir)[0]
print(model_name)
model_path = model_dir / model_name
print(model_path)

DESTINATION = base_dir / "stream_results.pkl"
print("SAVING TO", DESTINATION)

#%%
#%%
tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(model_path)
tf.get_logger().setLevel(logging.INFO)

# t_min = 0.4
# t_max = 0.95
# t_steps = int(np.ceil((t_max - t_min) / 0.05)) + 1
# threshs = np.linspace(t_min, t_max, t_steps).tolist()
flags = FlagTest(
    wav=str(base_dir / "stream.wav"),
    ground_truth=str(base_dir / "labels.txt"),
    target_keyword=target,
    # detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    #detection_thresholds=threshs,  # step threshold 0.05
    detection_thresholds=[0.65],  # step threshold 0.05
)
results = {}
results[target], inferences = calculate_streaming_accuracy(model, model_settings, flags)

with open(DESTINATION, "wb") as fh:
    pickle.dump(results, fh)
np.save(base_dir / "raw_inferences.npy", inferences)

#%%
merchant_video = Path("/home/mark/tinyspeech_harvard/merchant_video/")
merchant_wav = merchant_video / "stream.wav"
# t_min = 0.4
# t_max = 0.95
# t_steps = int(np.ceil((t_max - t_min) / 0.05)) + 1
# threshs = np.linspace(t_min, t_max, t_steps).tolist()
flags = FlagTest(
    wav=str(merchant_wav),
    ground_truth=str(merchant_video / "labels.txt"),
    target_keyword="merchant",
    # detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    detection_thresholds=[0.4],  # step threshold 0.05
)
model_settings = input_data.standard_microspeech_model_settings(3)
res = generate_inferences_offline(None, model_settings, flags)
res.shape
#%%
model_settings

#%%

# threshold=0.6
# flags = FlagTest(
#     wav=str(base_dir / "stream.wav"),
#     ground_truth=str(base_dir / "labels.txt"),
#     target_keyword=target,
#     detection_thresholds=[threshold]
# )
# inferences = np.load(base_dir / "inferences/inferences.npy")
# results={}
# results[target] = calculate_inferences_offline(inferences, flags)

# DESTINATION = base_dir / "results" / (f"stream_results_{threshold}.pkl")
# with open(DESTINATION, "wb") as fh:
#     pickle.dump(results, fh)


#%%
##############################
### VISUALIZE STREAM TIMELINE
##############################


def viz_stream_timeline(
    groundtruth,
    found_words,
    target,
    threshold,
    time_tolerance_ms=1500,
    num_nontarget_words=None,
):
    fig, ax = plt.subplots()
    gt_target_times = [t for g, t in groundtruth if g == target]
    print("gt target occurences", len(gt_target_times))
    found_target_times = [t for f, t in found_words if f == target]
    print("num found targets", len(found_target_times))

    # find false negatives
    false_negatives = 0
    for word, time in groundtruth:
        if word == target:
            ax.axvline(x=time, linestyle="--", linewidth=2, color=(0, 0.4, 0, 0.5))

            latest_time = time + time_tolerance_ms
            earliest_time = time - time_tolerance_ms
            potential_match = False
            for found_time in found_target_times:
                if found_time > latest_time:
                    break
                if found_time < earliest_time:
                    continue
                potential_match = True
            if not potential_match:
                false_negatives += 1
                ax.axvline(
                    x=time,
                    linestyle="-",
                    linewidth=4,
                    color=(0.8, 0.3, 0, 0.5),
                    ymax=0.9,
                )

    # find true/false positives
    false_positives = 0  # no groundtruth match for model-found word
    true_positives = 0
    for word, time in found_words:
        if word == target:
            # q = False
            # if time == 189540:
            #    q=True
            #    ax.axvline(x=time, linestyle="-", linewidth=3, color=(1,0,1,0.4), ymax=0.5)
            # else:
            ax.axvline(
                x=time, linestyle="-", linewidth=3, color=(0, 0, 1, 0.4), ymax=0.6
            )
            # highlight spurious words
            latest_time = time + time_tolerance_ms
            earliest_time = time - time_tolerance_ms
            potential_match = False
            for gt_time in gt_target_times:
                if gt_time > latest_time:
                    break
                if gt_time < earliest_time:
                    continue
                # if q:
                #    print(gt_time, np.abs(gt_time - time))
                #    print("b")
                potential_match = True
            if not potential_match:
                false_positives += 1
                ax.axvline(
                    x=time, linestyle="-", linewidth=5, color=(1, 0, 0, 0.4), ymax=0.4
                )
            else:
                true_positives += 1
    tpr = true_positives / len(gt_target_times)
    print("true positives ", true_positives, "TPR:", tpr)
    # TODO(MMAZ) is there a beter way to calculate false positive rate?
    # fpr = false_positives / (false_positives + true_negatives)
    # fpr = false_positives / negatives
    print(
        "false positives (model detection when no groundtruth target is present)",
        false_positives,
    )
    if num_nontarget_words is not None:
        fpr = false_positives / num_nontarget_words
        print(
            "FPR",
            f"{fpr:0.2f} {false_positives} false positives/{num_nontarget_words} nontarget words",
        )
        fpr_s = f"{fpr:0.2f}"
    else:
        fpr_s = "[not enough info]"

    max_x = groundtruth[-1][1] + 1000
    ax.set_xlim([0, max_x])
    # ax.set_xlim([175000,200000])
    fig.set_size_inches(40, 5)
    ax.set_title(
        f"{target} -> threshold: {threshold:0.2f}, TPR: {tpr}, FPR: {fpr_s}, false positives: {false_positives}, false_negatives: {false_negatives}, groundtruth positives: {len(gt_target_times)}"
    )
    # fig.savefig(f"/home/mark/tinyspeech_harvard/tmp/analysis/{target}/{target}_{threshold:0.2f}.png",dpi=300)
    return fig, ax


#%%
sse = "/home/mark/tinyspeech_harvard/streaming_sentence_experiments/"
res = sse + "old_merchant_5_shot/stream_results.pkl"
target = "merchant"
with open(res, "rb") as fh:
    results = pickle.load(fh)
for ix, thresh in enumerate(results[target].keys()):
    print(ix, thresh)

thresh_ix = 13
thresh, (stats, all_found_words) = list(results[target].items())[thresh_ix]
print("THRESH", thresh)
fig, ax = viz_stream_timeline(
    stats._gt_occurrence, all_found_words, target, thresh, num_nontarget_words=2328
)


#%%
############################################################################
#    Speech Commands
############################################################################
scanalysisdir = "/home/mark/tinyspeech_harvard/xfer_speechcommands_5/"
scmodeldir = scanalysisdir + "models/"
DESTINATION = scanalysisdir + "streaming_results_w_words.pkl"

scmodels = os.listdir(scmodeldir)
results = {}
for modelname in scmodels:
    tf.get_logger().setLevel(logging.ERROR)
    model = tf.keras.models.load_model(scmodeldir + modelname)
    tf.get_logger().setLevel(logging.INFO)

    # xfer_5_shot_6_epochs_marvin_val_acc_0.95
    prefix = "xfer_5_shot_6_epochs_"
    target = modelname[len(prefix) :].split("_")[0]
    print(target)

    flags = FlagTest(
        wav=f"{scanalysisdir}/streaming/{target}/streaming_test.wav",
        ground_truth=f"{scanalysisdir}/streaming/{target}/streaming_labels.txt",
        target_keyword=target,
        detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    )
    results[target] = calculate_streaming_accuracy(model, audio_dataset, FLAGS=flags)
# with open(DESTINATION, "wb") as fh:
#     pickle.dump(results, fh)

#%%
scanalysisdir = "/home/mark/tinyspeech_harvard/xfer_speechcommands_5/"
with open(scanalysisdir + "streaming_results_w_words.pkl", "rb") as fh:
    results = pickle.load(fh)

#%%
os.listdir(scmodeldir)

#%% analyze short tree
tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(
    scmodeldir + "xfer_5_shot_6_epochs_tree_val_acc_0.93"
)
tf.get_logger().setLevel(logging.INFO)

target = "tree"
flags = FlagTest(
    wav=f"{scanalysisdir}/small_streaming/small_{target}/small.wav",
    ground_truth=f"{scanalysisdir}/small_streaming/small_{target}/small.txt",
    target_keyword=target,
    detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
)
results = {}
results[target] = calculate_streaming_accuracy(model, model_settings, FLAGS=flags)

#%%
print("---")

#%%


#%%
# Why do the curves drop here? is Y-axis not the TPR? 
# is it possible this is happening instead?
# (number of ALL model positives, both true and false ones)/(# of groundtruth positives)
# seems to not be the case, i am using n_correct / n_targets
# need to dig in further to the stats behavior

fig = go.Figure()
for target, results_per_threshold in results.items():
    print(target)
    target_match_over_gt_positives = []
    false_positives_over_silence_or_unknown = []
    thresh_labels = []
    for thresh, (stats, found_words) in results_per_threshold.items():
        infomsg, statdict = stats.print_accuracy_stats()
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
        print(
            ":::", thresh, n_correct / n_targets, n_false_positives / n_silence_unknown
        )
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
# fig.write_html(scanalysisdir + "speech_commands_streaming_roc.html")
# fig.write_html(scanalysisdir + "short_tree.html")

fig


#%%  long pauses between also
############################################################################
modelname = "xfer_5_shot_6_epochs_also_val_acc_0.89"
model_path = f"/home/mark/tinyspeech_harvard/xfer_efnet_5/models/{modelname}"

tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(model_path)
tf.get_logger().setLevel(logging.INFO)

flags = FlagTest(
    wav=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/also2/streaming_test.wav",
    ground_truth=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/also2/streaming_labels.txt",
    target_keyword="also",
    detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    # detection_thresholds=[0.6],
)
results = {}
results["also"] = calculate_streaming_accuracy(model, audio_dataset, FLAGS=flags)


#%%


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
    wav=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/also3/streaming_test.wav",
    ground_truth=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/also3/streaming_labels.txt",
    target_keyword="also",
    # detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    detection_thresholds=[0.6],
)

# scanalysisdir="/home/mark/tinyspeech_harvard/xfer_speechcommands_5/"
# scmodeldir = scanalysisdir + "models/"
# tf.get_logger().setLevel(logging.ERROR)
# model = tf.keras.models.load_model(scmodeldir + 'xfer_5_shot_6_epochs_tree_val_acc_0.93')
# tf.get_logger().setLevel(logging.INFO)

# target="tree"
# FLAGS = FlagTest(
#     wav=f"{scanalysisdir}/small_streaming/small_{target}/small.wav",
#     ground_truth=f"{scanalysisdir}/small_streaming/small_{target}/small.txt",
#     target_keyword=target,
#     detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
# )

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
        int(np.ceil(audio_data_end / clip_stride_samples)),
        model_settings["spectrogram_length"],
        model_settings["fingerprint_width"],
    )
)
print("building spectrograms")
# Inference along audio stream.
for ix, audio_data_offset in enumerate(range(0, audio_data_end, clip_stride_samples)):
    input_start = audio_data_offset
    input_end = audio_data_offset + clip_duration_samples
    spectrograms[ix] = audio_dataset.to_micro_spectrogram(audio[input_start:input_end])

inferences = model.predict(spectrograms[:, :, :, np.newaxis])
print("inferences complete")

#%%

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
import accuracy_utils as mau

#%%
import importlib

importlib.reload(mau)


#%%

# for q in stats._gt_occurrence:
#     if q[1] > 1165640:
#         print(q)

FLAGS.time_tolerance_ms

#%%


threshold = 0.15
target = "also"
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
for ix, audio_data_offset in enumerate(range(0, audio_data_end, clip_stride_samples)):
    output_softmax = inferences[ix]
    current_time_ms = int(audio_data_offset * 1000 / sample_rate)
    recognize_commands.process_latest_result(
        output_softmax, current_time_ms, recognize_element
    )
    if (
        recognize_element.is_new_command
        and recognize_element.founded_command != "_silence_"
    ):
        # print(current_time_ms/1000, output_softmax, recognize_element.founded_command, recognize_element.score)
        # print(current_time_ms/1000, recognize_element.founded_command)#, recognize_element.score)
        all_found_words.append([recognize_element.founded_command, current_time_ms])
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
print("LENGTH", len(all_found_words))
# calculate final stats for full wav file:
stats.calculate_accuracy_stats(all_found_words, -1, FLAGS.time_tolerance_ms)
msg, info = stats.print_accuracy_stats()
statdict = info
n_targets = statdict["num_groundtruth_target"]
n_silence_unknown = statdict["num_groundtruth_unknown_or_silence"]
n_correct = statdict["matched"][target]
n_false_positives = (
    statdict["wrong"][input_data.SILENCE_LABEL]
    + statdict["wrong"][input_data.UNKNOWN_WORD_LABEL]
)
# target_match_over_gt_positives.append(n_correct / n_targets)
tpr = n_correct / n_targets
# false_positives_over_silence_or_unknown.append(
#     n_false_positives / n_silence_unknown
# )
fpr = n_false_positives / n_silence_unknown
# thresh_labels.append(f"thresh: {threshold}")
print(info)
print("threshold", threshold, "tpr", tpr, "fpr", fpr)
#%%
print(len(stats._gt_occurrence))
print(len(all_found_words))
num_unknowns = len([w for w in all_found_words if w[0] == "_unknown_"])
print(num_unknowns, num_unknowns / len(all_found_words))

#%%
print(
    {
        "correct_match_percentage": 88.7218045112782,
        "wrong_match_percentage": 7.518796992481203,
        "howmanyfp": 0,
        "howmanyfn": 7,
        "wrong": {"_silence_": 0, "_unknown_": 2, "also": 0},
        "matched": {"_silence_": 0, "_unknown_": 56, "also": 62},
        "num_groundtruth_target": 71,
        "num_groundtruth_unknown_or_silence": 62,
    }
)

#%%
afw = [["_unknown_", 2820], ["also", 16580], ["_unknown_", 24980], ["also", 25500]]
stats._gt_occurrence[:5]

#%%
len(stats._gt_occurrence)
#%%
gt_u, gt_t = [], []
for l, t in stats._gt_occurrence:
    if l == "_unknown_":
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
    if l == "_unknown_":
        f_u.append([t, 0.5])
        f_t.append([t, 0])
    else:
        f_u.append([t, 0])
        f_t.append([t, 0.5])
f_u = np.array(f_u)
f_t = np.array(f_t)

#%%
fig, ax = plt.subplots()
ax.plot(gt_u[:, 0], gt_u[:, 1], label="groundtruth unknown", color="blue")
ax.fill_between(gt_u[:, 0], gt_u[:, 1], color=(0, 0, 0.8, 0.3))
ax.plot(gt_t[:, 0], gt_t[:, 1], label="groundtruth target", color="green")
ax.fill_between(gt_t[:, 0], gt_t[:, 1], color=(0.2, 0.8, 0, 0.3))

ax.plot(f_u[:, 0], f_u[:, 1], label="found unknown", color="orange")
ax.fill_between(f_u[:, 0], f_u[:, 1], color="orange")
ax.plot(f_t[:, 0], f_t[:, 1], label="found target", color="purple")
ax.fill_between(f_t[:, 0], f_t[:, 1], color="purple")
ax.legend(loc="upper right")
fig.set_size_inches(30, 5)

