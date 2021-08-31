import os
from typing import Optional
import tempfile
from pathlib import Path
import shutil
import shlex
import subprocess
import json

import fire

from embedding import input_data
from embedding import batch_streaming_analysis as sa
from embedding.tpr_fpr import tpr_fpr


def main(
    keyword: str,
    modelpath: os.PathLike,
    wav: os.PathLike,
    groundtruth: Optional[os.PathLike] = None,
    transcript: Optional[os.PathLike] = None,
    visualizer: bool = False,
    serve_port: int = 8080,
    detection_threshold: float = 0.9,
    inference_chunk_len_seconds: int = 1200,
    language: str = "unspecified_language",
):
    """
    Runs inference on a streaming audio file. Example invocation:
      $ python -m embedding.run_inference --keyword mask --modelpath mask_model --wav mask_radio.wav 
    Args
      keyword: target keyword for few-shot KWS
      modelpath: path to the trained few-shot model
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
    
    """

    # create groundtruth if needed
    if groundtruth is None:
        fd, groundtruth = tempfile.mkstemp(prefix="empty_", suffix=".txt")
        os.close(fd)
        print(f"created {groundtruth}")
        created_temp_gt = True
    else:
        created_temp_gt = False

    assert os.path.exists(modelpath), f"{modelpath} inference model not found"
    assert os.path.exists(wav), f"{wav} streaming audio wavfile not found"
    assert Path(wav).suffix == ".wav", f"{wav} filetype not supported"
    assert (
        inference_chunk_len_seconds > 0
    ), "inference_chunk_len_seconds must be positive"

    print(f"performing inference using detection threshold {detection_threshold}")

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
    results = sa.eval_stream_test(streamtarget)

    detections_with_confidence = results[keyword][0][1][detection_threshold][1]

    for d in detections_with_confidence:
        print(d)

    # cleanup groundtruth if needed
    if created_temp_gt:
        os.remove(groundtruth)
        print(f"deleted {groundtruth}")

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

    for f in viz_files:
        if os.path.exists(f):
            print(f"ERROR {f} already exists")
            return

    # copy wav to serving destination
    shutil.copy2(wav, viz_wav)

    # write detections to json
    with open(viz_detections, 'w') as fh:
        json.dump(detections_with_confidence, fh)

    # create waveform visualization .dat file
    stream_dat_cmd = f"audiowaveform -i {wav} -o {viz_dat} -b 8"
    res = subprocess.check_output(args=shlex.split(stream_dat_cmd))
    print(res.decode("utf8"))

    # optionally copy transcript to serving destination
    if transcript is not None:
        assert os.path.exists(transcript), f"unable to read {transcript}"
        assert Path(transcript).suffix == ".json", f"transcript does not end in .json"
        assert os.path.exists(data_dest), f"unable to find {data_dest} from current directory"
        print(f"Copying transcript to {viz_transcript}")
        shutil.copy2(transcript, viz_transcript)
    
    # host the site 
    serve=f"npx serve --listen {serve_port} --no-clipboard visualizer/"
    proc = subprocess.Popen(args=shlex.split(serve))
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nTerminating visualization server")
        proc.terminate()
    
    for f in viz_files:
        if os.path.exists(f):
            print(f"deleting {f}")
            os.remove(f)


if __name__ == "__main__":
    fire.Fire(main)
