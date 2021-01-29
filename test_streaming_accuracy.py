# based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/test_streaming_accuracy.py


#%%
import argparse
from dataclasses import dataclass
import logging
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
import input_data

sys.path.insert(
    0, "/home/mark/tinyspeech_harvard/tensorflow/tensorflow/examples/speech_commands/"
)
from accuracy_utils import StreamingAccuracyStats
from recognize_commands import RecognizeCommands, RecognizeResult


#%%

TARGET_KEYWORD_LABEL = "always"


@dataclass(frozen=True)
class FlagTest:
    wav = "/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/streaming_test.wav"
    ground_truth = (
        "/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/streaming_labels.txt"
    )
    labels = [
        input_data.SILENCE_LABEL,
        input_data.UNKNOWN_WORD_LABEL,
        TARGET_KEYWORD_LABEL,
    ]
    clip_duration_ms = 1000
    clip_stride_ms = 20  # window_stride_ms in model_settings
    average_window_duration_ms = 500
    detection_threshold = 0.6
    suppression_ms = 500
    time_tolerance_ms = 1500


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

#%% MAKE A SPECTROGRAM AND PREDICT
clipdir = "/home/mark/tinyspeech_harvard/frequent_words/en/clips/always/"
fs = os.listdir(clipdir)
c = clipdir + np.random.choice(fs)

(audio, sample_rate) = tf.audio.decode_wav(tf.io.read_file(c), desired_channels=1)
audio = audio.numpy().flatten()
audio.shape

spec = audio_dataset.to_micro_spectrogram(audio)
print(spec.shape)
s = spec.numpy()
model.predict(s[np.newaxis, :, : np.newaxis])

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


#%% CALCULATE STREAMING ACCURACY 
FLAGS = FlagTest()
# model =
# audio_dataset =
wav_loader = tf.io.read_file(FLAGS.wav)
(audio, sample_rate) = tf.audio.decode_wav(wav_loader, desired_channels=1)
sample_rate = sample_rate.numpy()
audio = audio.numpy().flatten()

recognize_commands = RecognizeCommands(
    labels=FLAGS.labels,
    average_window_duration_ms=FLAGS.average_window_duration_ms,
    detection_threshold=FLAGS.detection_threshold,
    suppression_ms=FLAGS.suppression_ms,
    minimum_count=4,
)

# Init instance of StreamingAccuracyStats and load ground truth.
stats = StreamingAccuracyStats()
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
print('building spectrograms')
# Inference along audio stream.
for ix, audio_data_offset in enumerate(range(0, audio_data_end, clip_stride_samples)):
    input_start = audio_data_offset
    input_end = audio_data_offset + clip_duration_samples
    spectrograms[ix] = audio_dataset.to_micro_spectrogram(audio[input_start:input_end])

inferences = model.predict(spectrograms[:, :, :, np.newaxis])
print('inferences complete')

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
        all_found_words.append([recognize_element.founded_command, current_time_ms])
        stats.calculate_accuracy_stats(
            all_found_words, current_time_ms, FLAGS.time_tolerance_ms
        )
        recognition_state = stats.delta()
        print(
            "{}ms {}:{}{}".format(
                current_time_ms,
                recognize_element.founded_command,
                recognize_element.score,
                recognition_state,
            )
        )
        stats.print_accuracy_stats()
print("DONE")
stats.calculate_accuracy_stats(all_found_words, -1, FLAGS.time_tolerance_ms)
stats.print_accuracy_stats()
# # markstats(FLAGS, stats)

#%%
stats.print_accuracy_stats()
#%%
stats._which_matched
#%%
stats._which_wrong
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_streaming_accuracy")
    parser.add_argument(
        "--wav", type=str, default="", help="The wave file path to evaluate."
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="",
        help="The ground truth file path corresponding to wav file.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="The label file path containing all possible classes.",
    )
    parser.add_argument(
        "--model", type=str, default="", help="The model used for inference"
    )
    parser.add_argument(
        "--input-names",
        type=str,
        nargs="+",
        default=["decoded_sample_data:0", "decoded_sample_data:1"],
        help="Input name list involved in model graph.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="labels_softmax:0",
        help="Output name involved in model graph.",
    )
    parser.add_argument(
        "--clip-duration-ms",
        type=int,
        default=1000,
        help="Length of each audio clip fed into model.",
    )
    parser.add_argument(
        "--clip-stride-ms",
        type=int,
        default=30,
        help="Length of audio clip stride over main trap.",
    )
    parser.add_argument(
        "--average_window_duration_ms",
        type=int,
        default=500,
        help="Length of average window used for smoothing results.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.7,
        help="The confidence for filtering unreliable commands",
    )
    parser.add_argument(
        "--suppression_ms",
        type=int,
        default=500,
        help="The time interval between every two adjacent commands",
    )
    parser.add_argument(
        "--time-tolerance-ms",
        type=int,
        default=1500,
        help="Time tolerance before and after the timestamp of this audio clip "
        "to match ground truth",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Whether to print streaming accuracy on stdout.",
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)

