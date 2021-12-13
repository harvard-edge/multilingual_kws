# based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/test_streaming_accuracy.py

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
from typing import List, Optional

import numpy as np
import tensorflow as tf


import multilingual_kws.embedding.input_data as input_data
from multilingual_kws.embedding.accuracy_utils import StreamingAccuracyStats
from multilingual_kws.embedding.single_target_recognize_commands import (
    SingleTargetRecognizeCommands,
    RecognizeResult,
)

# %%
@dataclass(frozen=True)
class StreamFlags:
    wav: os.PathLike
    ground_truth: os.PathLike
    target_keyword: str
    detection_thresholds: List[float]
    clip_duration_ms: int = 1000
    clip_stride_ms: int = 20  # window_stride_ms in model_settings
    # average_window_duration_ms: int = 500
    average_window_duration_ms: int = 100
    suppression_ms: int = 500
    time_tolerance_ms: int = 750
    minimum_count: int = 4
    max_chunk_length_sec: int = 1200  # max chunk length is 1200 seconds (20 minutes) by default

    def labels(self) -> List[str]:
        return [
            input_data.SILENCE_LABEL,
            input_data.UNKNOWN_WORD_LABEL,
            self.target_keyword,
        ]


def calculate_streaming_accuracy(
    model, model_settings, flag_list, existing_inferences=None
):
    assert len(set([f.wav for f in flag_list])) == 1, "can only process one wav"
    assert len(set([f.clip_duration_ms for f in flag_list])) == 1, "cannot vary"
    assert len(set([f.clip_stride_ms for f in flag_list])) == 1, "cannot vary"
    wav = flag_list[0].wav
    wav_loader = tf.io.read_file(wav)
    (audio, sample_rate) = tf.audio.decode_wav(wav_loader, desired_channels=1)
    sample_rate = sample_rate.numpy()
    audio = audio.numpy().flatten()

    # Init instance of StreamingAccuracyStats and load ground truth.
    # number of samples in the entire audio file
    data_samples = audio.shape[0]
    # number of samples in one utterance (e.g., a 1 second clip)
    clip_duration_samples = int(flag_list[0].clip_duration_ms * sample_rate / 1000)
    # number of samples in one stride
    clip_stride_samples = int(flag_list[0].clip_stride_ms * sample_rate / 1000)
    # leave space for one full clip at end
    audio_data_end = data_samples - clip_duration_samples

    chunks = []
    max_chunk_len_samples = (
        flag_list[0].max_chunk_length_sec * sample_rate
    )  # max chunk length in samples

    if data_samples < max_chunk_len_samples:  # this check might be unnecessary
        chunks.append(audio)
    else:
        for ind, offset in enumerate(range(0, data_samples, max_chunk_len_samples)):
            if (
                offset + max_chunk_len_samples > data_samples
            ):  # add extra check to make sure last chunk isn't too short?
                chunks.append(audio[offset : offset + max_chunk_len_samples])
            else:
                chunks.append(audio[offset:])

    if existing_inferences is not None:
        inferences = existing_inferences
    else:
        start = True
        for chunk in chunks:
            chunk_len = chunk.shape[0]
            chunk_data_end = chunk_len - clip_duration_samples

            # num spectrograms: in the range expression below, if there is a remainder
            # in audio_data_end/clip_stride_samples, we need one additional slot
            # for a spectrogram
            spectrograms = np.zeros(
                (
                    int(np.ceil(chunk_data_end / clip_stride_samples)),
                    model_settings["spectrogram_length"],
                    model_settings["fingerprint_width"],
                )
            )
            print("building spectrograms", flush=True)
            # Inference along audio stream.
            for ix, audio_data_offset in enumerate(
                range(0, chunk_data_end, clip_stride_samples)
            ):
                input_start = audio_data_offset
                input_end = audio_data_offset + clip_duration_samples
                spectrograms[ix] = input_data.to_micro_spectrogram(
                    model_settings, chunk[input_start:input_end]
                )

            inferences_ = model.predict(spectrograms[:, :, :, np.newaxis])

            if start:
                inferences = inferences_
                start = False
            else:
                inferences = np.concatenate((inferences, inferences_))

    results = []
    for FLAGS in flag_list:
        res_thresh = {}
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
            print(f"results for {threshold:0.2f}")
            # calculate final stats for full wav file:
            stats.calculate_accuracy_stats(all_found_words, -1, FLAGS.time_tolerance_ms)
            stats.print_accuracy_stats()
            res_thresh[threshold] = (all_found_words, all_found_words_w_confidences)
        results.append((FLAGS, res_thresh))
    return results, inferences


# %%

# %%


@dataclass
class StreamTarget:
    target_lang: str
    target_word: str
    model_path: os.PathLike
    stream_flags: StreamFlags
    destination_result_pkl: Optional[os.PathLike] = None
    destination_result_inferences: Optional[os.PathLike] = None


def eval_stream_test(st: StreamTarget, live_model=None):
    if live_model is not None:
        model = live_model
    else:
        tf.get_logger().setLevel(logging.ERROR)
        model = tf.keras.models.load_model(st.model_path)
        tf.get_logger().setLevel(logging.INFO)

    model_settings = input_data.standard_microspeech_model_settings(label_count=3)

    if st.destination_result_pkl is not None and os.path.isfile(
        st.destination_result_pkl
    ):
        print("results already present", st.destination_result_pkl, flush=True)
        return
    inferences_exist = False
    if st.destination_result_inferences is not None:
        if os.path.isfile(st.destination_result_inferences):
            print("inferences already present", flush=True)
            loaded_inferences = np.load(st.destination_result_pkl)
            inferences_exist = True

    results = {}
    if inferences_exist:
        results[st.target_word], _ = calculate_streaming_accuracy(
            model, model_settings, st.stream_flags, loaded_inferences
        )
    else:
        results[st.target_word], inferences = calculate_streaming_accuracy(
            model, model_settings, st.stream_flags
        )

    if st.destination_result_pkl is not None:
        print("SAVING results TO\n", st.destination_result_pkl)
        with open(st.destination_result_pkl, "wb") as fh:
            pickle.dump(results, fh)
    if not inferences_exist and st.destination_result_inferences is not None:
        print(
            "SAVING inferences TO\n", st.destination_result_inferences, flush=True
        )
        np.save(st.destination_result_inferences, inferences)

    # https://keras.io/api/utils/backend_utils/
    tf.keras.backend.clear_session()
    return results


def batch_streaming_analysis():
    batch_data_to_process = []

    # fmt: off
    # sse = Path("/home/mark/tinyspeech_harvard/paper_data/streaming_batch_sentences/")
    # sse = Path("/home/mark/tinyspeech_harvard/paper_data/ooe_streaming_batch_sentences/")
    # sse = Path("/home/mark/tinyspeech_harvard/paper_data/streaming_batch_perword/")
    # sse = Path("/home/mark/tinyspeech_harvard/paper_data/ooe_streaming_batch_perword/")

    # for silence-padded data:
    # dest_dir = Path("/home/mark/tinyspeech_harvard/paper_data/results_streaming_batch_sentences/")
    # dest_dir = Path("/home/mark/tinyspeech_harvard/paper_data/results_ooe_streaming_batch_sentences/")
    # dest_dir = Path("/home/mark/tinyspeech_harvard/paper_data/results_streaming_batch_perword/")
    # dest_dir = Path("/home/mark/tinyspeech_harvard/paper_data/results_ooe_streaming_batch_perword/")

    # for context-padded data:
    # dest_dir = Path("/home/mark/tinyspeech_harvard/paper_data/context_results_streaming_batch_sentences/")
    # dest_dir = Path("/home/mark/tinyspeech_harvard/paper_data/context_results_ooe_streaming_batch_sentences/")
    # dest_dir = Path("/home/mark/tinyspeech_harvard/paper_data/context_results_streaming_batch_perword/")
    # dest_dir = Path("/home/mark/tinyspeech_harvard/paper_data/context_results_ooe_streaming_batch_perword/")
    # fmt: on

    for ix, lang_dir in enumerate(os.listdir(sse)):
        if not os.path.isdir(sse / lang_dir):
            continue  # skip the data generator shellscript and the logfiles
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

            flags = StreamFlags(
                wav=str(stream_wav),
                ground_truth=str(stream_label),
                target_keyword=target_word,
                detection_thresholds=np.linspace(
                    0.05, 1, 20
                ).tolist(),  # step threshold 0.05
            )
            d = StreamTarget(
                target_lang=target_lang,
                target_word=target_word,
                model_path=model_path,
                destination_result_pkl=destination_result_pkl,
                destination_result_inferences=destination_result_inferences,
                stream_flags=flags,
            )
            batch_data_to_process.append(d)

    np.random.shuffle(batch_data_to_process)

    n_wavs = len(batch_data_to_process)
    print("n wavs", n_wavs, flush=True)

    batchdata_file = (
        "/home/mark/tinyspeech_harvard/paper_data/data_ooe_streaming_batch_perword.pkl"
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
# python embedding/batch_streaming_analysis.py > /home/mark/tinyspeech_harvard/paper_data/results_ooe_multilang_sentence.log
# python embedding/batch_streaming_analysis.py > /home/mark/tinyspeech_harvard/paper_data/results_ooe_multilang_perword.log
if __name__ == "__main__":
    batch_streaming_analysis()
