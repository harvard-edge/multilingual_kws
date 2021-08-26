import os
from typing import Optional
import tempfile
from pathlib import Path

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
      detection_threshold: confidence threshold for inference (default=0.9)
      inference_chunk_len_seconds: inference will first chunk the wavfile into portions
        in order to avoid exhausting GPU memory. this sets the size of each chunk. 
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


if __name__ == "__main__":
    fire.Fire(main)
