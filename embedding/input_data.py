import tensorflow as tf
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)
from tensorflow.python.platform import gfile
import numpy as np  # TODO(mmaz) tf2.4 np from tf.experimental
import os
import glob
import math
from pathlib import Path
from dataclasses import dataclass

SILENCE_LABEL = "_silence_"
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = "_unknown_"
UNKNOWN_WORD_INDEX = 1


def to_micro_spectrogram(model_settings, audio):
    sample_rate = model_settings["sample_rate"]
    window_size_ms = (model_settings["window_size_samples"] * 1000) / sample_rate
    window_step_ms = (model_settings["window_stride_samples"] * 1000) / sample_rate
    int16_input = tf.cast(tf.multiply(audio, 32768), tf.int16)
    # https://git.io/Jkuux
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
    return output


def file2spec(model_settings, filepath):
    """there's a version of this that adds bg noise in AudioDataset"""
    audio_binary = tf.io.read_file(filepath)
    audio, _ = tf.audio.decode_wav(
        audio_binary,
        desired_channels=1,
        desired_samples=model_settings["desired_samples"],
    )
    audio = tf.squeeze(audio, axis=-1)
    return to_micro_spectrogram(model_settings, audio)


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


def standard_microspeech_model_settings(label_count: int):
    return prepare_model_settings(
        label_count=label_count,
        sample_rate=16000,
        clip_duration_ms=1000,
        window_size_ms=30,
        window_stride_ms=20,
        feature_bin_count=40,
        preprocess="micro",
    )


def add_background(foreground_audio, background_audio, background_volume):
    foreground_rms = tf.sqrt(tf.reduce_mean(tf.square(foreground_audio)))
    background_rms = tf.sqrt(tf.reduce_mean(tf.square(background_audio)))

    snr_scaling = tf.cond(
        tf.greater(background_rms, tf.constant(0.0)),
        lambda: tf.divide(foreground_rms, background_rms),
        lambda: tf.constant(0.0),
    )

    # background_data_scaled has the same average signal power (really, rms) as the foreground_audio
    background_data_scaled = tf.multiply(background_audio, snr_scaling)

    # reduce the scaled background volume to the desired volume
    bg_mul = tf.multiply(background_data_scaled, background_volume)
    bg_add = tf.add(bg_mul, foreground_audio)
    return tf.clip_by_value(bg_add, -1.0, 1.0)


@dataclass(frozen=True)
class SpecAugParams:
    percentage: float = 80.0
    # how many augmentations to include, inclusive
    frequency_n_range: int = 2
    # how large each mask should be (pixels)
    frequency_max_px: int = 2
    # how many augmentations to include, inclusive
    time_n_range: int = 2
    # how large each mask should be (pixels)
    time_max_px: int = 2


class AudioDataset:
    def __init__(
        self,
        model_settings,
        commands,
        background_data_dir,
        unknown_files,
        time_shift_ms=100,
        background_frequency=0.8,
        background_volume_range=0.1,
        silence_percentage=10.0,
        unknown_percentage=10.0,
        spec_aug_params=SpecAugParams(),
        seed=None,
    ) -> None:
        self.get_background_data(background_data_dir)
        self.model_settings = model_settings
        self.max_time_shift_samples = self.timeshift_samples(
            time_shift_ms=time_shift_ms
        )
        self.background_frequency = background_frequency  # freq. between 0-1
        self.background_volume_range = background_volume_range

        # below list prepending is order-sensitive (unknown, then silence)
        # so that with both, labels are always ordered:
        # [silence, unknown, word1, word2,...]
        self.unknown_percentage = unknown_percentage
        self.unknown_files = unknown_files
        if len(self.unknown_files) > 0 and self.unknown_percentage > 0:
            commands = [UNKNOWN_WORD_LABEL] + commands
        self.silence_percentage = silence_percentage  # pct between 0-100
        if self.silence_percentage > 0:
            commands = [SILENCE_LABEL] + commands
        self.commands = tf.convert_to_tensor(commands)

        self.spec_aug_params = spec_aug_params

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
        # TODO(mmaz) why is this reshape necessary?
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

    def get_unknown(self):
        unknown_index = self.gen.uniform([], 0, len(self.unknown_files), dtype=tf.int32)
        file_path = tf.gather(self.unknown_files, unknown_index)
        audio_binary = tf.io.read_file(file_path)
        waveform = self.decode_audio(audio_binary)
        return waveform

    def augment(self, audio, label):
        # see discussion on random:
        # https://github.com/tensorflow/tensorflow/issues/35682#issuecomment-574092770

        audio = (
            self.random_timeshift(audio) if self.max_time_shift_samples > 0 else audio
        )
        if self.gen.uniform([], 0, 1) < (self.silence_percentage / 100):
            background_volume = self.gen.uniform([], 0, 1)
            label = SILENCE_LABEL
            audio = self.random_background_sample(background_volume)
        elif len(self.unknown_files) > 0 and self.gen.uniform([], 0, 1) < (
            self.unknown_percentage / 100
        ):
            audio = self.get_unknown()
            audio = (
                self.random_timeshift(audio)
                if self.max_time_shift_samples > 0
                else audio
            )
            # TODO(mmaz): add in background noise?
            label = UNKNOWN_WORD_LABEL
        # mix in background?
        elif self.gen.uniform([], 0, 1) < self.background_frequency:
            background_volume = self.gen.uniform([], 0, self.background_volume_range)
            background_audio = self.random_background_sample()
            audio = add_background(audio, background_audio, background_volume)
        return audio, label

    def spec_augment(self, spectrogram):
        # https://git.io/JLvGB
        # https://arxiv.org/pdf/1904.08779.pdf

        s = tf.shape(spectrogram)
        # e.g., 49x40
        # cannot unpack in one line (OperatorNotAllowedInGraphError:
        #   iterating over `tf.Tensor` is not allowed in Graph execution)
        time_max = s[0]
        freq_max = s[1]

        freq_n = self.gen.uniform(
            [], 0, self.spec_aug_params.frequency_n_range + 1, dtype=tf.int32
        )
        time_n = self.gen.uniform(
            [], 0, self.spec_aug_params.time_n_range + 1, dtype=tf.int32
        )

        @tf.function
        def freq_body(ix, spectrogram_aug):
            size = self.gen.uniform(
                [], 1, self.spec_aug_params.frequency_max_px + 1, dtype=tf.int32
            )
            start = self.gen.uniform([], 0, freq_max - size, dtype=tf.int32)
            mask = tf.concat(
                [
                    tf.ones([time_max, start], dtype=tf.float32),
                    tf.zeros([time_max, size], dtype=tf.float32),
                    tf.ones([time_max, freq_max - start - size], dtype=tf.float32),
                ],
                axis=1,
            )
            return ix + 1, tf.multiply(spectrogram_aug, mask)

        spectrogram = tf.while_loop(
            lambda ix, augmented: ix < freq_n, freq_body, (0, spectrogram)
        )[1]

        @tf.function
        def time_body(ix, spectrogram_aug):
            size = self.gen.uniform(
                [], 1, self.spec_aug_params.time_max_px + 1, dtype=tf.int32
            )
            start = self.gen.uniform([], 0, time_max - size, dtype=tf.int32)
            mask = tf.concat(
                [
                    tf.ones([start, freq_max], dtype=tf.float32),
                    tf.zeros([size, freq_max], dtype=tf.float32),
                    tf.ones([time_max - start - size, freq_max], dtype=tf.float32),
                ],
                axis=0,
            )
            return ix + 1, tf.multiply(spectrogram_aug, mask)

        spectrogram = tf.while_loop(
            lambda ix, augmented: ix < time_n, time_body, (0, spectrogram)
        )[1]

        return spectrogram

    def map_spec_aug(self, spectrogram, label_id):
        if self.gen.uniform([], 0, 1) < (self.spec_aug_params.percentage / 100):
            spectrogram = self.spec_augment(spectrogram)
        return spectrogram, label_id

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

    def get_single_target_waveforms(self, file_path):
        label = self.commands[-1]
        audio_binary = tf.io.read_file(file_path)
        waveform = self.decode_audio(audio_binary)
        return waveform, label

    def get_spectrogram_and_label_id(self, audio, label):
        micro_spec = to_micro_spectrogram(self.model_settings, audio)
        lc = label == self.commands
        label_id = tf.argmax(lc)
        return micro_spec, label_id

    def get_label_id_from_filename(self, filepath):
        # only for rebalancer
        label = self.get_label(filepath)
        lc = label == self.commands
        label_id = tf.argmax(lc)
        return label_id

    def add_channel(self, spectrogram, label_id):
        """from width x height to width x height x channel"""
        return tf.expand_dims(spectrogram, -1), label_id

    def file2spec_w_bg(self, filepath):
        audio_binary = tf.io.read_file(filepath)
        waveform = self.decode_audio(audio_binary)
        waveform = self._add_bg(waveform)
        return to_micro_spectrogram(self.model_settings, waveform)

    def _add_bg(self, audio):
        background_volume = self.gen.uniform([], 0, self.background_volume_range)
        background_audio = self.random_background_sample()
        return add_background(audio, background_audio, background_volume)

    def init_single_target(self, AUTOTUNE, files, is_training):
        """assumes a single-target model, reads label from self.commands"""
        files_ds = tf.data.Dataset.from_tensor_slices(files)

        # buffer size with shuffle: https://stackoverflow.com/a/48096625
        waveform_ds = files_ds.map(
            self.get_single_target_waveforms, num_parallel_calls=AUTOTUNE
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
        spectrogram_ds = (
            spectrogram_ds.map(self.map_spec_aug, num_parallel_calls=AUTOTUNE)
            if is_training
            else spectrogram_ds
        )

        return spectrogram_ds.map(self.add_channel, num_parallel_calls=AUTOTUNE)

    def init_from_parent_dir(self, AUTOTUNE, files, is_training):
        """uses the parent dir as the label name"""
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        # TODO(mmaz) should we rebalance here?
        # if rebalance:
        # we havent generated any unknowns or silence here
        # so we should probably just rebalance naively
        # 1/len(commands without silence/unknown)
        # non_word_pct = self.silence_percentage, self.unknown_percentage
        # word_pct = (1-non_word_pct) / (len(self.commands))
        # target_dist = [non_word_pct] + [ for c in self.commands]
        # resampler = tf.data.experimental.rejection_resample(
        #     self.get_label_id_from_filename, target_dist=target_dist
        # )
        # train_ds = files_ds.apply(resampler)

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
        spectrogram_ds = (
            spectrogram_ds.map(self.map_spec_aug, num_parallel_calls=AUTOTUNE)
            if is_training
            else spectrogram_ds
        )

        return spectrogram_ds.map(self.add_channel, num_parallel_calls=AUTOTUNE)

    def _random_silence(self):
        background_volume = self.gen.uniform([], 0, 1)
        label = SILENCE_LABEL
        audio = self.random_background_sample(background_volume)
        return audio, label

    def _random_unknown(self):
        audio = self.get_unknown()
        label = UNKNOWN_WORD_LABEL
        return audio, label

    def _random_silence_unknown(self, n_files):
        n_silent = int(n_files * self.silence_percentage / 100)
        n_unknown = int(n_files * self.unknown_percentage / 100)
        silence_samples = tf.data.Dataset.range(n_silent).map(
            lambda _: self._random_silence()
        )
        unknown_samples = tf.data.Dataset.range(n_unknown).map(
            lambda _: self._random_unknown()
        )
        return silence_samples.concatenate(unknown_samples)

    def eval_with_silence_unknown(self, AUTOTUNE, files, label_from_parent_dir: bool):
        """includes silence + unknown
        Args:
          label_from_parent_dir: bool
            if True, uses the parent directory as the label name
            if False, assumes a single-target model and reads label from self.commands
        """
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        if label_from_parent_dir:
            waveform_ds = files_ds.map(
                self.get_waveform_and_label, num_parallel_calls=AUTOTUNE
            )
        else:
            assert (
                self.commands.shape[0] == 3
            ), "model does not support both silence and unknown"
            waveform_ds = files_ds.map(
                self.get_single_target_waveforms, num_parallel_calls=AUTOTUNE
            )

        waveform_ds = waveform_ds.concatenate(self._random_silence_unknown(len(files)))
        spectrogram_ds = waveform_ds.map(
            self.get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE
        )
        return spectrogram_ds.map(self.add_channel, num_parallel_calls=AUTOTUNE)


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

    wdir = "/home/mark/tinyspeech_harvard/frequent_words/en/clips/"
    unknown_words = []
    for w in ["group", "green", "great"]:
        wavs = glob.glob(wdir + w + "/*.wav")
        unknown_words += wavs
    a = AudioDataset(model_settings, commands, bg_datadir, unknown_words=unknown_words)

    # x = a.random_timeshift(x)

    # from scipy.io.wavfile import write
    # write(f"tmp/bg.wav", model_settings["sample_rate"], x.numpy())

    # AUTOTUNE = tf.data.experimental.AUTOTUNE
    # files_ds = tf.data.Dataset.from_tensor_slices([f])
    # waveform_ds = files_ds.map(a.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    # waveform_ds = waveform_ds.map(a.augment, num_parallel_calls=AUTOTUNE)
    # rs = waveform_ds.repeat().take(10)
    # for ix, (audio, label) in enumerate(rs):
    #     print(ix, label)
    #     write(f"tmp/example{ix}.wav", model_settings["sample_rate"], audio.numpy())

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
