# %%
from dataclasses import asdict
import os
import glob
import shutil
import subprocess
import time
from collections import Counter
import csv
import pickle
import datetime
from pathlib import Path
import pprint
from typing import List, Optional, Tuple

import numpy as np
import sox
import pydub
from pydub.playback import play
import textgrid

from embedding import word_extraction, transfer_learning
from embedding import batch_streaming_analysis as sa
import input_data
from luganda_info import WavTranscript

# %%
# generate nontarget data
l_data = Path("/media/mark/hyperion/makerere/uliza-clips")
l_csv = l_data / "transcripts.csv"

l_test = Path("/media/mark/hyperion/makerere/alignment/cs288/")
l_clips = l_test / "cs288_clips"

workdir = Path("/home/mark/tinyspeech_harvard/luganda")
os.makedirs(workdir / "cs288_eval" / "nontarget", exist_ok=True)
dest_wavfile = str(workdir / "cs288_eval" / "nontarget" / "nontarget_stream.wav")
groundtruth_data = workdir / "cs288_eval" / "nontarget" / "nontarget_groundtruth.pkl"
assert not os.path.isfile(dest_wavfile), "already exists"
assert not os.path.isfile(groundtruth_data), "already exists"

NUM_WAVS = 160

# select random wavs without keyword to intersperse stream with
keywords_to_exclude = set(["corona", "korona", "kolona", "coronavirus", "mask", "masiki", "masks" ])
non_targets = []
with open(l_csv, "r") as fh:
    reader = csv.reader(fh)
    next(reader)  # skip header [wav_filename,wav_filesize,transcript]
    for ix, row in enumerate(reader):
        wav_filename = l_data / row[0]
        assert os.path.isfile(wav_filename), "wav not found"
        transcript = row[2]
        has_kw = any([w in keywords_to_exclude for w in transcript.split()])
        if not has_kw:
            non_targets.append(WavTranscript(wav=wav_filename, transcript=transcript))
ntlog = set()

total_wav_duration_s = 0.0
to_combine = []

stream_data = []

for ix in range(NUM_WAVS):
    non_target_ix = np.random.randint(len(non_targets))
    ntlog.add(non_target_ix)
    non_target = non_targets[non_target_ix]

    nontarget_duration_s = sox.file_info.duration(non_target.wav)

    total_wav_duration_s += nontarget_duration_s

    to_combine.append(str(non_target.wav))

    nd = asdict(non_target)
    nd["duration_s"] = nontarget_duration_s
    stream_data.append(nd)

groundtruth = dict(
    groundtruth_target_times_ms=[],
    stream_data=stream_data,
    keyword_wav_transcripts=[],
    non_targets=non_targets,
    ixs=[],
)

print("duration in minutes", total_wav_duration_s / 60)

combiner = sox.Combiner()
combiner.convert(samplerate=16000, n_channels=1)
combiner.build(to_combine, dest_wavfile, "concatenate")

with open(groundtruth_data, "wb") as fh:
    pickle.dump(groundtruth, fh)

# %%
groundtruth_data = workdir / "cs288_eval" / "nontarget" / "nontarget_groundtruth.pkl"
with open(groundtruth_data, 'rb') as fh:
    groundtruth = pickle.load(fh)

for stream_segment in groundtruth["stream_data"]:
    transcript = stream_segment["transcript"]
    for w in transcript.split():
        if w in keywords_to_exclude:
            raise ValueError("keyword present", transcript)

# %%
# NONTARGET DATA
# corona:
# results for 0.80
# No ground truth yet, 52false positives
# results for 0.85
# No ground truth yet, 33false positives
# results for 0.90
# No ground truth yet, 23false positives
# results for 0.95
# No ground truth yet, 7false positives
#
# mask:
# results for 0.75
# No ground truth yet, 30false positives
# results for 0.80
# No ground truth yet, 19false positives
# results for 0.85
# No ground truth yet, 11false positives
# results for 0.90
# No ground truth yet, 5false positives
# results for 0.95
# No ground truth yet, 3false positives
# results for 1.00
# No ground truth yet, 0false positives
#
# --------------------------------------------------------
# TARGET DATA
# corona:
# results for 0.60
# No ground truth yet, 211false positives
# results for 0.65
# No ground truth yet, 173false positives
# results for 0.70
# No ground truth yet, 145false positives
# results for 0.75
# No ground truth yet, 115false positives
# results for 0.80
# No ground truth yet, 95false positives
# results for 0.85
# No ground truth yet, 78false positives
# results for 0.90
# No ground truth yet, 45false positives
# results for 0.95
# No ground truth yet, 20false positives
# results for 1.00
# No ground truth yet, 0false positives
# 0.7999999999999999
# {'false_accepts_per_hour': 117.74101540216778,
#  'false_negatives': 31,
#  'false_positives': 43,
#  'false_rejections_per_instance': 0.37349397590361444,
#  'fpr': 0.014266755142667552,
#  'groundtruth_positives': 83,
#  'thresh': 0.7999999999999999,
#  'tpr': 0.6265060240963856,
#  'true_positives': 52}
# 0.85
# {'false_accepts_per_hour': 82.14489446662867,
#  'false_negatives': 35,
#  'false_positives': 30,
#  'false_rejections_per_instance': 0.42168674698795183,
#  'fpr': 0.009953550099535502,
#  'groundtruth_positives': 83,
#  'thresh': 0.85,
#  'tpr': 0.5783132530120482,
#  'true_positives': 48}
# 0.9
# {'false_accepts_per_hour': 38.33428408442672,
#  'false_negatives': 52,
#  'false_positives': 14,
#  'false_rejections_per_instance': 0.6265060240963856,
#  'fpr': 0.0046449900464499,
#  'groundtruth_positives': 83,
#  'thresh': 0.9,
#  'tpr': 0.37349397590361444,
#  'true_positives': 31}
# 0.95
# {'false_accepts_per_hour': 8.214489446662867,
#  'false_negatives': 66,
#  'false_positives': 3,
#  'false_rejections_per_instance': 0.7951807228915663,
#  'fpr': 0.0009953550099535502,
#  'groundtruth_positives': 83,
#  'thresh': 0.95,
#  'tpr': 0.20481927710843373,
#  'true_positives': 17}
#
# mask:
# results for 0.70
# No ground truth yet, 122false positives
# results for 0.75
# No ground truth yet, 108false positives
# results for 0.80
# No ground truth yet, 93false positives
# results for 0.85
# No ground truth yet, 75false positives
# results for 0.90
# No ground truth yet, 64false positives
# results for 0.95
# No ground truth yet, 44false positives
# results for 1.00
# No ground truth yet, 0false positives
# 0.7999999999999999
# {'false_accepts_per_hour': 66.7943180300129,
#  'false_negatives': 24,
#  'false_positives': 25,
#  'false_rejections_per_instance': 0.26373626373626374,
#  'fpr': 0.008015389547932029,
#  'groundtruth_positives': 91,
#  'thresh': 0.7999999999999999,
#  'tpr': 0.7472527472527473,
#  'true_positives': 68}
# 0.85
# {'false_accepts_per_hour': 42.74836353920826,
#  'false_negatives': 32,
#  'false_positives': 16,
#  'false_rejections_per_instance': 0.3516483516483517,
#  'fpr': 0.005129849310676499,
#  'groundtruth_positives': 91,
#  'thresh': 0.85,
#  'tpr': 0.6483516483516484,
#  'true_positives': 59}
# 0.9
# {'false_accepts_per_hour': 24.045954490804647,
#  'false_negatives': 36,
#  'false_positives': 9,
#  'false_rejections_per_instance': 0.3956043956043956,
#  'fpr': 0.0028855402372555306,
#  'groundtruth_positives': 91,
#  'thresh': 0.9,
#  'tpr': 0.6043956043956044,
#  'true_positives': 55}
# 0.95
# {'false_accepts_per_hour': 2.6717727212005165,
#  'false_negatives': 48,
#  'false_positives': 1,
#  'false_rejections_per_instance': 0.5274725274725275,
#  'fpr': 0.0003206155819172812,
#  'groundtruth_positives': 91,
#  'thresh': 0.95,
#  'tpr': 0.4725274725274725,
#  'true_positives': 43}