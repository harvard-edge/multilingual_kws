import os
from typing import Optional, List, Dict
import tempfile
from pathlib import Path
import shutil
import shlex
import subprocess
import multiprocessing
import json
import glob
import csv

import fire

from multilingual_kws.embedding import input_data
from multilingual_kws.embedding import batch_streaming_analysis as sa
from multilingual_kws.embedding import transfer_learning
from multilingual_kws.embedding.tpr_fpr import tpr_fpr, get_groundtruth


def eval(streamtarget: sa.StreamTarget, results: Dict):
    results.update(sa.eval_stream_test(streamtarget))


def inference(
    keywords: List[str],
    modelpaths: str,
    wav: os.PathLike,
    groundtruth: Optional[os.PathLike] = None,
    transcript: Optional[os.PathLike] = None,
    visualizer: bool = False,
    serve_port: int = 8080,
    detection_threshold: float = 0.9,
    inference_chunk_len_seconds: int = 1200,
    language: str = "unspecified_language",
    write_detections: Optional[os.PathLike] = None,
    overwrite: bool = False,
):
    """
    Runs inference on a streaming audio file. Example invocation:
      $ python -m embedding.run_inference --keyword mask --modelpath mask_model --wav mask_radio.wav
    Args
      keyword: target keywords for few-shot KWS (pass in as [word1, word2, word3])
      modelpaths: comma-demlimited list of paths to finetuned few-shot models
      wav: path to the audio file to analyze
      groundtruth: optional path to a groundtruth audio file
      transcript: optional path to a groundtruth transcript (for data visualization)
      visualizer: run the visualization server after performing inference
      serve_port: browser port to run visualization server on
      detection_threshold: confidence threshold for inference (default=0.9)
      inference_chunk_len_seconds: we chunk the wavfile into portions
        to avoid exhausting GPU memory - this sets the chunksize.
        default = 1200 seconds (i.e., 20 minutes)
      language: target language (for data visualization)
      write_detections: path to save detections.json
      overwrite: preserves (and overwrites) visualization outputs
    """

    if len(keywords[0]) == 1:
        print(f"NOTE - assuming a single keyword was passed in: {keywords}")
        keywords = [keywords]
    print(f"Target keywords: {keywords}")

    modelpaths = modelpaths.split(",")
    assert len(modelpaths) == len(
        set(keywords)
    ), f"discrepancy: {len(modelpaths)} modelpaths provided for {len(set(keywords))} keywords"

    # create groundtruth if needed
    if groundtruth is None:
        fd, groundtruth = tempfile.mkstemp(prefix="empty_", suffix=".txt")
        os.close(fd)
        print(f"created {groundtruth}")
        created_temp_gt = True
    else:
        created_temp_gt = False

    for p in modelpaths:
        assert os.path.exists(p), f"{p} inference model not found"
    assert os.path.exists(wav), f"{wav} streaming audio wavfile not found"
    assert Path(wav).suffix == ".wav", f"{wav} filetype not supported"
    assert (
        inference_chunk_len_seconds > 0
    ), "inference_chunk_len_seconds must be positive"

    print(f"performing inference using detection threshold {detection_threshold}")

    unsorted_detections = []
    for keyword, modelpath in zip(keywords, modelpaths):
        flags = sa.StreamFlags(
            wav=wav,
            ground_truth=groundtruth,
            target_keyword=keyword,
            detection_thresholds=[detection_threshold],
            average_window_duration_ms=100,
            suppression_ms=500,
            time_tolerance_ms=750,  # only used when graphing
            max_chunk_length_sec=inference_chunk_len_seconds,
        )
        streamtarget = sa.StreamTarget(
            target_lang=language,
            target_word=keyword,
            model_path=modelpath,
            stream_flags=[flags],
        )
        manager = multiprocessing.Manager()
        results = manager.dict()
        # TODO(mmaz): note that the summary tpr/fpr calculated within eval is incorrect when multiple
        # targets are being evaluated - groundtruth_labels.txt contains multiple targets but
        # each model is only single-target (at the moment)
        p = multiprocessing.Process(target=eval, args=(streamtarget, results))
        p.start()
        p.join()

        unsorted_detections.extend(results[keyword][0][1][detection_threshold][1])

    detections_with_confidence = list(sorted(unsorted_detections, key=lambda d: d[1]))

    for d in detections_with_confidence:
        print(d)

    # cleanup groundtruth if needed
    if created_temp_gt:
        os.remove(groundtruth)
        print(f"deleted {groundtruth}")
        # no groundtruth
        detections_with_confidence = [
            dict(keyword=d[0], time_ms=d[1], confidence=d[2], groundtruth="ng")
            for d in detections_with_confidence
        ]
    else:
        # modify detections using groundtruth
        groundtruth_data = []
        with open(groundtruth, "r") as fh:
            reader = csv.reader(fh)
            for row in reader:
                groundtruth_data.append((row[0], float(row[1])))

        detections_with_confidence = get_groundtruth(
            detections_with_confidence, keywords, groundtruth_data
        )

    detections = dict(
        keywords=keywords,
        detections=detections_with_confidence,
        min_threshold=detection_threshold,
    )

    # write detections to json
    if write_detections is not None:
        with open(write_detections, "w") as fh:
            json.dump(detections, fh)

    if not visualizer:
        return

    print("running visualizer")
    data_dest = Path("visualizer/data")

    assert os.path.isdir(data_dest), f"{data_dest} not found"

    viz_dat = data_dest / "stream.dat"
    viz_transcript = data_dest / "full_transcript.json"
    viz_wav = data_dest / "stream.wav"
    viz_detections = data_dest / "detections.json"
    viz_files = [viz_dat, viz_transcript, viz_wav, viz_detections]

    if not overwrite:
        for f in viz_files:
            if os.path.exists(f):
                print(f"ERROR {f} already exists")
                return

    # copy wav to serving destination
    shutil.copy2(wav, viz_wav)

    # write detections to json (this time, for viz)
    with open(viz_detections, "w") as fh:
        json.dump(detections, fh)

    # create waveform visualization .dat file
    stream_dat_cmd = f"audiowaveform -i {wav} -o {viz_dat} -b 8"
    res = subprocess.check_output(args=shlex.split(stream_dat_cmd))
    print(res.decode("utf8"))

    # optionally copy transcript to serving destination
    if transcript is not None:
        assert os.path.exists(transcript), f"unable to read {transcript}"
        assert Path(transcript).suffix == ".json", f"transcript does not end in .json"
        assert os.path.exists(
            data_dest
        ), f"unable to find {data_dest} from current directory"
        print(f"Copying transcript to {viz_transcript}")
        shutil.copy2(transcript, viz_transcript)

    # host the site
    serve = f"npx serve --listen {serve_port} --no-clipboard visualizer/"
    proc = subprocess.Popen(args=shlex.split(serve))
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nTerminating visualization server")
        proc.terminate()

    if not overwrite:
        for f in viz_files:
            if os.path.exists(f):
                print(f"deleting {f}")
                os.remove(f)


def train(
    keyword: str,
    samples_dir: os.PathLike,
    embedding: os.PathLike,
    unknown_words: os.PathLike,
    background_noise: os.PathLike,
    output: os.PathLike,
    num_epochs: int = 4,
    num_batches: int = 1,
    primary_learning_rate: float = 0.001,
    batch_size: int = 64,
    unknown_percentage: float = 50.0,
    base_model_output: str = "dense_2",
):
    """Fine-tune few-shot model from embedding representation. The embedding
    representation and unknown words can be downloaded from
    https://github.com/harvard-edge/multilingual_kws/releases
    The background noise directory can be downloaded from:
    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

    Args:
      keyword: target keyword
      samples_dir: directory of 1-second 16KHz target .wav samples
      embedding: path to embedding representation
      unknown_words: path to unknown words directory
      background_noise: path to Google Speech Commands background noise directory
      output: modelname for saving the model (specified as a path)
      num_epochs: number of finetuning epochs
      num_batches: number of finetuning batches
      primary_learning_rate: finetuning LR
      batch_size: batch size
      unknown_percentage: percentage of samples to draw from unknown_words
      base_model_output: layer to use for embedding representation
    """

    assert (
        Path(background_noise).name == "_background_noise_"
    ), f"only tested with GSC _background_noise_ directory, please provide a path {background_noise}"

    for d in [samples_dir, embedding, unknown_words, background_noise]:
        assert os.path.isdir(d), f"directory {d} not found"

    if os.path.exists(output):
        print(f"Warning: overwriting {output}")

    samples = glob.glob(samples_dir + os.path.sep + "*.wav")
    assert len(samples) > 0, "no sample .wavs found"
    for s in samples:
        cmd = f"soxi {s}"
        res = subprocess.check_output(shlex.split(cmd))
        out = res.decode("utf8")
        checks = ["75 CDDA sectors", "16000 samples", "00:00:01.00"]

        if not all([c in out for c in checks]):
            raise ValueError(
                f"{s} appears to not be a 16KHz 1-second wav file according to soxi \n{out}"
            )

    print(f"{len(samples)} training samples found:\n" + "\n".join(samples))

    uftxt = "unknown_files.txt"
    unknown_words = Path(unknown_words)
    assert os.path.isfile(unknown_words / uftxt), f"{unknown_words/uftxt} not found"
    unknown_files = []
    with open(unknown_words / uftxt, "r") as fh:
        for w in fh.read().splitlines():
            unknown_files.append(str(unknown_words / w))

    print("Training model")
    model_settings = input_data.standard_microspeech_model_settings(3)
    name, model, details = transfer_learning.transfer_learn(
        target=keyword,
        train_files=samples,
        val_files=samples,
        unknown_files=unknown_files,
        num_epochs=num_epochs,
        num_batches=num_batches,
        batch_size=batch_size,
        primary_lr=primary_learning_rate,
        backprop_into_embedding=False,
        embedding_lr=0,
        model_settings=model_settings,
        base_model_path=embedding,
        base_model_output=base_model_output,
        UNKNOWN_PERCENTAGE=unknown_percentage,
        csvlog_dest=None,
    )
    print(f"saving model to {output}")
    model.save(output)


if __name__ == "__main__":
    fire.Fire(dict(inference=inference, train=train))
