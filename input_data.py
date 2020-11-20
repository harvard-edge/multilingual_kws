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

SILENCE_LABEL = "_silence_"
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = "_unknown_"
UNKNOWN_WORD_INDEX = 1

def _next_power_of_two(x):
    """Calculates the smallest enclosing power of two for an input.
    source: https://git.io/JkuvF

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
    """
    source: https://git.io/JkuvF
    Calculates common settings needed for all models.

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


def add_background(foreground_audio, background_audio, background_volume):
    # TODO(mmaz): should we downscale the foreground audio first, to avoid clipping?
    bg_mul = tf.multiply(background_audio, background_volume)
    bg_add = tf.add(bg_mul, foreground_audio)
    return tf.clip_by_value(bg_add, -1.0, 1.0)


class AudioDataset:
    def __init__(
        self,
        model_settings,
        commands,
        background_data_dir,
        time_shift_ms=100,
        background_frequency=0.8,
        background_volume_range=0.1,
        enable_silence=True,
        silence_percentage=10.0,
        enable_unknown=False,
        unknown_percentage=0,
        seed=None,
    ) -> None:
        self.get_background_data(background_data_dir)
        self.model_settings = model_settings
        self.max_time_shift_samples = self.timeshift_samples(
            time_shift_ms=time_shift_ms
        )
        self.background_frequency = background_frequency  # freq. between 0-1
        self.background_volume_range = background_volume_range

        self.enable_silence = enable_silence
        if self.enable_silence:
            commands = [SILENCE_LABEL] + commands
        self.silence_percentage = silence_percentage  # pct between 0-100
        self.enable_unknown = enable_unknown
        if self.enable_unknown or unknown_percentage > 0:
            commands = [UNKNOWN_WORD_LABEL] + commands
            raise NotImplementedError
        self.commands = tf.convert_to_tensor(commands)

        if seed:
            self.gen = tf.random.Generator.from_seed(seed)
        else:
            self.gen = tf.random.Generator.from_non_deterministic_state()

    def timeshift_samples(self, time_shift_ms=100):
        sample_rate = self.model_settings["sample_rate"]
        return int((time_shift_ms * sample_rate) / 1000)

    #####
    ## augmentations
    #####

    # todo: import more augmentations from https://github.com/mozilla/DeepSpeech/blob/3762a9b5884d4646c223198f89400265d1d50fae/training/deepspeech_training/util/augmentations.py
    # https://github.com/mozilla/DeepSpeech/blob/master/doc/TRAINING.rst#sample-domain-augmentations
    # overlay (bgaudio), reverb, pitch, frequency/time masking

    def random_background_sample(self, background_volume=1.0):
        desired_samples = self.model_settings["desired_samples"]
        background_index = tf.random.uniform(
            [], 0, self.background_sizes.shape[0], tf.dtypes.int32
        )
        wav_length = self.background_sizes[background_index]
        background_samples = self.background_data[background_index, 0:wav_length]

        background_offset = self.gen.uniform(
            [], 0, wav_length - desired_samples, dtype=tf.int32,
        )
        background_clipped = background_samples[
            background_offset : (background_offset + desired_samples)
        ]
        background_clipped = tf.multiply(background_clipped, background_volume)
        #TODO(mmaz) why is this reshape necessary?
        return tf.reshape(background_clipped, (desired_samples,))

    def random_timeshift(self, audio):
        desired_samples = self.model_settings["desired_samples"]
        time_shift_amount = self.gen.uniform(
            [],
            -self.max_time_shift_samples,
            self.max_time_shift_samples,
            dtype=tf.int32,
        )
        if time_shift_amount > 0:
            # pad beginning of wav
            # padding = tf.constant(([[time_shift_amount, 0]]))
            padding = tf.expand_dims(
                tf.stack([time_shift_amount, tf.constant(0)]), axis=0
            )
            offset = 0
        else:
            # padding = tf.constant([[0, -time_shift_amount]])
            padding = tf.expand_dims(
                tf.stack([tf.constant(0), -time_shift_amount]), axis=0
            )
            offset = -time_shift_amount
        pad = tf.pad(audio, padding, mode="CONSTANT")
        sliced = tf.slice(pad, [offset], [desired_samples])
        return sliced

    def augment(self, audio, label):
        # see discussion on random:
        # https://github.com/tensorflow/tensorflow/issues/35682#issuecomment-574092770

        audio = (
            self.random_timeshift(audio) if self.max_time_shift_samples > 0 else audio
        )
        if (
            self.enable_silence
            and self.gen.uniform([], 0, 1) < self.silence_percentage / 100
        ):
            background_volume = self.gen.uniform([], 0, 1)
            label = SILENCE_LABEL
            audio = self.random_background_sample(background_volume)
            return audio, label
        elif self.gen.uniform([], 0, 1) < self.background_frequency:
            background_volume = self.gen.uniform([], 0, self.background_volume_range)
            background_audio = self.random_background_sample()
            audio = add_background(audio, background_audio, background_volume)
        # # elif self.enable_unknown and self.gen.uniform(0,1) < self.unknown_percentage/100:
        # #     audio = get_unknown()
        # #     label=UNKNOWN_WORD_LABEL
        # #     raise NotImplementedError
        return audio, label

    #####
    ## -----end augmentations
    #####

    def get_background_data(self, background_dir):
        # TODO(mmaz): could not figure out how to use ragged tensors for this
        # so instead hacked together a single padded array :(
        background_data = []
        background_sizes = []
        for wav_path in gfile.Glob(os.path.join(background_dir, "*.wav")):
            wav_binary = tf.io.read_file(wav_path)
            audio, _ = tf.audio.decode_wav(wav_binary, desired_channels=1)
            background_sizes.append(audio.shape[0])
            background_data.append(tf.squeeze(audio, axis=-1))
        # build padded array
        bgdata = np.zeros(
            (len(background_sizes), max(background_sizes)), dtype=np.float32
        )
        for i in range(bgdata.shape[0]):
            wav = background_data[i].numpy()
            length = wav.shape[0]
            bgdata[i, 0:length] = wav
        self.background_data = tf.convert_to_tensor(bgdata)
        self.background_sizes = tf.convert_to_tensor(background_sizes)

    def decode_audio(self, audio_binary):
        desired_samples = self.model_settings["desired_samples"]
        audio, _ = tf.audio.decode_wav(
            audio_binary, desired_channels=1, desired_samples=desired_samples
        )
        return tf.squeeze(audio, axis=-1)

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2]

    def get_waveform_and_label(self, file_path):
        label = self.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = self.decode_audio(audio_binary)
        return waveform, label

    def to_micro_spectrogram(self, audio):
        sample_rate = self.model_settings["sample_rate"]
        window_size_ms = (
            self.model_settings["window_size_samples"] * 1000
        ) / sample_rate
        window_step_ms = (
            self.model_settings["window_stride_samples"] * 1000
        ) / sample_rate
        int16_input = tf.cast(tf.multiply(audio, 32768), tf.int16)
        # https://git.io/Jkuux
        micro_frontend = frontend_op.audio_microfrontend(
            int16_input,
            sample_rate=sample_rate,
            window_size=window_size_ms,
            window_step=window_step_ms,
            num_channels=self.model_settings["fingerprint_width"],
            out_scale=1,
            out_type=tf.float32,
        )
        output = tf.multiply(micro_frontend, (10.0 / 256.0))
        return output

    def get_spectrogram_and_label_id(self, audio, label):
        micro_spec = self.to_micro_spectrogram(audio)
        micro_spec = tf.expand_dims(micro_spec, -1)
        lc = label == self.commands
        label_id = tf.argmax(lc)
        return micro_spec, label_id

    def init(self, AUTOTUNE, files, is_training):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        # buffer size with shuffle: https://stackoverflow.com/a/48096625
        waveform_ds = files_ds.map(
            self.get_waveform_and_label, num_parallel_calls=AUTOTUNE
        )
        # https://www.tensorflow.org/tutorials/images/data_augmentation#apply_the_preprocessing_layers_to_the_datasets
        waveform_ds = (
            waveform_ds.map(self.augment, num_parallel_calls=AUTOTUNE)
            if is_training
            else waveform_ds
        )
        spectrogram_ds = waveform_ds.map(
            self.get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE
        )
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
    commands = ["ever"]
    bg_datadir = (
        "/home/mark/tinyspeech_harvard/frequent_words/en/clips/_background_noise_/"
    )

    a = AudioDataset(model_settings, commands, bg_datadir)

    # x = a.random_timeshift(x)
    # write(f"tmp/bg.wav", model_settings["sample_rate"], x.numpy())

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices([f])
    waveform_ds = files_ds.map(a.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    waveform_ds = waveform_ds.map(a.augment, num_parallel_calls=AUTOTUNE)
    rs = waveform_ds.repeat().take(10)
    for ix, (audio, label) in enumerate(rs):
        print(ix, label)
        write(f"tmp/example{ix}.wav", model_settings["sample_rate"], audio.numpy())
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    f_ds = a.init(AUTOTUNE, [f], is_training=True)
    for ix, (audio, label) in enumerate(f_ds.repeat().take(10)):
        print(label)
        print(audio.dtype, audio.shape)


    """
    f = tf.io.read_file(f)
    a = decode_audio(model_settings, f)

    m = timeshift_samples(model_settings)
    s = random_timeshift(model_settings, a, m)

    bgdata = get_background_data(
        ,
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
    """

    print("done")


if __name__ == "__main__":
    # test_timeshift()
    test()
