import tensorflow as tf
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)
from tensorflow.python.framework.op_def_registry import get
from tensorflow.python.platform import gfile
import numpy as np  # TODO(mmaz) tf2.4 np from tf.experimental
import os
import glob
import math
from scipy.io.wavfile import write

# from tensorflow.python.ops import gen_audio_ops as audio_ops


def _next_power_of_two(x):
    """Calculates the smallest enclosing power of two for an input.

  Args:
    x: Positive float or integer number.

  Returns:
    Next largest power of two integer.
  """
    return 1 if x == 0 else 2 ** (int(x) - 1).bit_length()


def prepare_model_settings(
    label_count,
    sample_rate,
    clip_duration_ms,
    window_size_ms,
    window_stride_ms,
    feature_bin_count,
    preprocess,
):
    """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    feature_bin_count: Number of frequency bins to use for analysis.
    preprocess: How the spectrogram is processed to produce features.

  Returns:
    Dictionary containing common settings.

  Raises:
    ValueError: If the preprocessing mode isn't recognized.
  """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = desired_samples - window_size_samples
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    if preprocess == "average":
        fft_bin_count = 1 + (_next_power_of_two(window_size_samples) / 2)
        average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
        fingerprint_width = int(math.ceil(fft_bin_count / average_window_width))
    elif preprocess == "mfcc":
        average_window_width = -1
        fingerprint_width = feature_bin_count
    elif preprocess == "micro":
        average_window_width = -1
        fingerprint_width = feature_bin_count
    else:
        raise ValueError(
            'Unknown preprocess mode "%s" (should be "mfcc",'
            ' "average", or "micro")' % (preprocess)
        )
    fingerprint_size = fingerprint_width * spectrogram_length
    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "spectrogram_length": spectrogram_length,
        "fingerprint_width": fingerprint_width,
        "fingerprint_size": fingerprint_size,
        "label_count": label_count,
        "sample_rate": sample_rate,
        "preprocess": preprocess,
        "average_window_width": average_window_width,
    }


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def decode_audio(model_settings, audio_binary):
    desired_samples = model_settings["desired_samples"]
    audio, _ = tf.audio.decode_wav(
        audio_binary, desired_channels=1, desired_samples=desired_samples
    )
    return tf.squeeze(audio, axis=-1)


def timeshift_samples(model_settings, time_shift_ms=100):
    sample_rate = model_settings["sample_rate"]
    return int((time_shift_ms * sample_rate) / 1000)


def random_timeshift(model_settings, audio, max_time_shift_samples):
    desired_samples = model_settings["desired_samples"]
    time_shift_amount = np.random.randint(
        -max_time_shift_samples, max_time_shift_samples
    )
    if time_shift_amount > 0:
        # pad beginning of wav
        padding = tf.constant(([[time_shift_amount, 0]]))
        offset = 0
    else:
        padding = tf.constant([[0, -time_shift_amount]])
        offset = -time_shift_amount
    pad = tf.pad(audio, padding, mode="CONSTANT")
    sliced = tf.slice(pad, [offset], [desired_samples])
    return sliced


def add_background(foreground_audio, background_audio, background_volume):
    # TODO(mmaz): should we downscale the foreground audio first, to avoid clipping?
    bg_mul = tf.multiply(background_audio, background_volume)
    bg_add = tf.add(bg_mul, foreground_audio)
    return tf.clip_by_value(bg_add, -1.0, 1.0)


def get_background_data(background_dir):
    background_data = []
    for wav_path in gfile.Glob(os.path.join(background_dir, "*.wav")):
        print(wav_path)
        wav_binary = tf.io.read_file(wav_path)
        audio, _ = tf.audio.decode_wav(wav_binary, desired_channels=1)
        background_data.append(tf.squeeze(audio, axis=-1))
    return background_data


def random_background_sample(model_settings, background_data):
    desired_samples = model_settings["desired_samples"]
    background_index = np.random.randint(len(background_data))
    background_samples = background_data[background_index]
    print(background_samples.shape[0])
    background_offset = np.random.randint(
        0, background_samples.shape[0] - model_settings["desired_samples"]
    )
    background_clipped = background_samples[
        background_offset : (background_offset + desired_samples)
    ]
    return background_clipped


def to_micro_spectrogram(model_settings, audio):
    # spectrogram = audio_ops.audio_spectrogram(
    #     audio,
    #     window_size=model_settings["window_size_samples"],
    #     stride=model_settings["window_stride_samples"],
    #     magnitude_squared=True,
    # )
    sample_rate = model_settings["sample_rate"]
    window_size_ms = (model_settings["window_size_samples"] * 1000) / sample_rate
    window_step_ms = (model_settings["window_stride_samples"] * 1000) / sample_rate
    int16_input = tf.cast(tf.multiply(audio, 32768), tf.int16)
    micro_frontend = frontend_op.audio_microfrontend(
        int16_input,
        sample_rate=sample_rate,
        window_size=window_size_ms,
        window_step=window_step_ms,
        num_channels=model_settings["fingerprint_width"],
        out_scale=1,
        out_type=tf.float32,
    )
    output = tf.multiply(micro_frontend, (10.0 / 256.0))
    return micro_frontend


def get_waveform_and_label(model_settings, file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(model_settings, audio_binary)
    return waveform, label


def get_spectrogram_and_label_id(model_settings, commands, audio, label):
    micro_spec = to_micro_spectrogram(model_settings, audio)
    micro_spec = tf.expand_dims(micro_spec, -1)
    lc = label == commands
    label_id = tf.argmax(lc)
    return micro_spec, label_id


def to_dataset(model_settings, commands, AUTOTUNE, files):
    # AUTOTUNE = tf.data.experimental.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    wfn = lambda file_path: get_waveform_and_label(model_settings, file_path)
    waveform_ds = files_ds.map(wfn, num_parallel_calls=AUTOTUNE)
    sfn = lambda wf, l: get_spectrogram_and_label_id(model_settings, commands, wf, l)
    spectrogram_ds = waveform_ds.map(sfn, num_parallel_calls=AUTOTUNE)
    return spectrogram_ds


def test_timeshift():
    length = 20
    a = tf.constant(range(length)) + 1
    print(a)
    ms = 4
    amt = np.random.randint(-ms, ms)
    print("amt", amt)
    if amt > 0:
        padding = tf.constant([[amt, 0]])
        offset = 0
    else:
        padding = tf.constant([[0, -amt]])
        offset = -amt

    pad = tf.pad(a, padding, mode="CONSTANT")
    print(pad)
    sliced = tf.slice(pad, [offset], [length])
    print(sliced)


def test():
    model_settings = prepare_model_settings(
        label_count=100,
        sample_rate=16000,
        clip_duration_ms=1000,
        window_size_ms=30,
        window_stride_ms=20,
        feature_bin_count=40,
        preprocess="micro",
    )
    f = "/home/mark/tinyspeech_harvard/frequent_words/en/clips/ever/common_voice_en_106360.wav"
    f = tf.io.read_file(f)
    a = decode_audio(model_settings, f)

    m = timeshift_samples(model_settings)
    s = random_timeshift(model_settings, a, m)

    bgdata = get_background_data(
        "/home/mark/tinyspeech_harvard/frequent_words/en/clips/_background_noise_/",
    )
    print("bgdata", len(bgdata))
    bg = random_background_sample(model_settings, bgdata)
    # silence
    silence = tf.multiply(bg, np.random.uniform(0, 1))
    # mixed with foreground
    background_volume_range = 0.1
    background_volume = np.random.uniform(0, background_volume_range)
    a_w_bg = add_background(s, bg, background_volume)

    micro = to_micro_spectrogram(model_settings, a_w_bg)

    # write("tmp/example.wav", model_settings["sample_rate"], a_w_bg.numpy())

    print("done")


if __name__ == "__main__":
    # test_timeshift()
    test()
