#%%
import datetime
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy.io import wavfile
import sox

import sys

from accuracy_utils import StreamingAccuracyStats

#%%


def make_frame(words_times, detections, clip_inferences, threshold):
    frame_width = 1920
    frame_height = 1080

    # detection_ixs = set(detections.keys())

    NUM_WORDS = 6
    assert len(words_times) <= NUM_WORDS, "too many words"
    if len(words_times) < NUM_WORDS:
        to_add = NUM_WORDS - len(words_times)
        words_times = to_add * [("", -1, -1)] + words_times
        # detection_ixs = {d + to_add for d in detection_ixs}
        # detections = {k + to_add: v for k, v in detections.items()}

    frame = np.zeros((frame_height, frame_width, 3), np.uint8)

    x_border = 50
    cv.rectangle(
        frame,
        pt1=(x_border, 100),
        pt2=(frame_width - x_border, 600),
        color=(255, 255, 255),
        thickness=4,
    )

    font = cv.FONT_HERSHEY_DUPLEX
    cv.putText(
        frame,
        f"Transcription [smoothed confidence] at threshold {threshold:0.2f}",
        (x_border, 90),
        font,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2,
        lineType=cv.LINE_AA,
    )

    word_boxwidth_px = (frame_width - 2 * x_border) // NUM_WORDS

    TEXT_HEIGHT = 375
    # also the number of 'blocks' for each character
    MAX_WORDLEN_TO_CENTER = 21

    for ix in range(NUM_WORDS):
        top_left = x_border + (ix * word_boxwidth_px), 200
        bottom_right = x_border + (ix + 1) * word_boxwidth_px, 500
        wordcolor = (255, 255, 255)
        (start_s, end_s) = words_times[ix][1], words_times[ix][2]
        det_score = None
        for d in detections:
            det_s = d["time_s"]
            if end_s < det_s:
                continue
            if start_s > det_s + 1:
                continue
            if start_s < 0:
                continue
            # within detection window
            cv.rectangle(
                frame, pt1=top_left, pt2=bottom_right, color=(0, 255, 0), thickness=-1,
            )
            wordcolor = (0, 0, 0)
            det_score = d["smoothed_score"]
        w = words_times[ix][0]

        word_size_px, _ = cv.getTextSize(w, font, fontScale=1, thickness=2)
        word_width_px = word_size_px[0]
        # need to skip to beginning of box
        word_start_px = top_left[0] + (word_boxwidth_px // 2) - (word_width_px // 2)

        cv.putText(
            frame,
            w,
            (word_start_px, TEXT_HEIGHT),
            font,
            fontScale=1,
            color=wordcolor,
            thickness=2,
            lineType=cv.LINE_AA,
        )

        if det_score is not None:
            cv.putText(
                frame,
                f"[{det_score:0.2f}]",
                (word_start_px, TEXT_HEIGHT + 50),
                font,
                fontScale=1,
                color=wordcolor,
                thickness=2,
                lineType=cv.LINE_AA,
            )

    ######## inference scores
    score_colors = [(255, 50, 55), (0, 10, 255), (0, 255, 0)]
    start_scores_height_px = 730
    start_scores_left_border = 400
    category_width_px = 220
    score_height_px = 60
    bottom_border = 10
    for ix, (category, score) in enumerate(clip_inferences.items()):
        # category label
        label_bottom = (
            start_scores_height_px + (ix + 1) * score_height_px - bottom_border
        )
        cv.rectangle(
            frame,
            pt1=(
                start_scores_left_border,
                start_scores_height_px + ix * score_height_px,
            ),
            pt2=(start_scores_left_border + category_width_px, label_bottom,),
            color=(200, 200, 200),
            thickness=-1,
        )
        cv.putText(
            frame,
            category,
            (start_scores_left_border, label_bottom - 12),
            font,
            fontScale=1.3,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv.LINE_AA,
        )

        # score background
        score_bg_width = 700
        score_left = start_scores_left_border + category_width_px
        score_top = start_scores_height_px + ix * score_height_px
        cv.rectangle(
            frame,
            pt1=(score_left, score_top,),
            pt2=(score_left + score_bg_width, label_bottom),
            color=(255, 255, 255),
            thickness=-1,
        )

        # score bar
        score_width = int(score * score_bg_width)
        cv.rectangle(
            frame,
            pt1=(score_left, score_top),
            pt2=(score_left + score_width, label_bottom),
            color=score_colors[ix],
            thickness=-1,
        )
    return frame


#clip_inferences = dict(silence=0.023, unknown=0.4, merchant=0.8)
clip_inferences = dict()

# detections = {
#     0: 0.6,
#     2: 0.928215,
# }
d1 = dict(target="merchant", smoothed_score=0.3, time_s=0.9)
d2 = dict(target="merchant", smoothed_score=0.1, time_s=2.8)
# words = "test word six words blah blah".split()
words_times = [
    ("test", 0, 0.5),
    ("word", 0.5, 1),
    ("six", 1, 2),
    ("blah", 2, 2.5),
    ("blah", 2.5, 2.7),
    ("blah", 2.7, 3),
]
# words_times = [("test", 0, 0.5)]
frame = make_frame(words_times, [d1, d2], clip_inferences, threshold=0.4)
# frame = make_frame(dict(), [], dict(), 0.4)
fig, ax = plt.subplots()
ax.imshow(frame)
fig.set_size_inches(20, 20)

#%%

#fmt: off
#video_dir = Path("/home/mark/tinyspeech_harvard/merchant_video/")
#transcription_file = "/home/mark/tinyspeech_harvard/merchant_video/full_transcription.pkl"
#wav_file = "/home/mark/tinyspeech_harvard/merchant_video/stream.wav"
#video_dir = Path("/home/mark/tinyspeech_harvard/multilingual_streaming_sentence_experiments/merchant/")
target_word = "iaith"
video_dir = Path("/home/mark/tinyspeech_harvard/streaming_sentence_experiments/") / target_word
transcription_file = video_dir / "full_transcription.pkl"
wav_file = video_dir / "stream.wav"
#fmt: on

frames_dir = video_dir / "frames"
assert os.path.isdir(frames_dir), "no frames dir"

with open(transcription_file, "rb") as fh:
    transcription = pickle.load(fh)

wav_duration = sox.file_info.duration(wav_file)

transcription[0], wav_duration

with open(video_dir / "stream_results.pkl", "rb") as fh:
    results = pickle.load(fh)


# thresh_ix = 13  # 8, 13
thresh_ix = 4
threshold, (_, _, all_found_words_w_scores) = list(results[target_word].items())[
    thresh_ix
]
print(threshold)

raw_inferences = np.load(video_dir / "raw_inferences.npy")
print(raw_inferences.shape)
#%%


#%%


def get_words(transcription, last_ix, frame_time_s):
    if last_ix >= len(transcription):
        return [w for w in transcription[-6:]], last_ix
    _, _, w_end = transcription[last_ix]
    while frame_time_s > w_end:
        last_ix += 1
        _, _, w_end = transcription[last_ix]
    # slice into only the last six each time
    return [w for w in transcription[:last_ix][-6:]], last_ix


def get_detections(found, words_times, last_ix_found):
    if len(words_times) == 0 or len(found) < last_ix_found:
        return dict(), last_ix_found
    start_time_s = words_times[0][1]
    end_time_s = words_times[-1][2]

    next_detection_s = found[last_ix_found][0]
    # still before the next word
    if next_detection_s > end_time_s:
        return dict(), last_ix_found

    while next_detection_s < start_time_s:
        last_ix_found += 1
        next_detection_s = found[last_ix_found][0]

    # return detections within the window
    d_ix = last_ix_found
    detections = []
    while found[d_ix][0] <= end_time_s:
        detections.append(dict(smoothed_score=found[d_ix][1], time_s=found[d_ix][0]))
        d_ix += 1
    return detections, last_ix_found


# FPS seems a little off, words appear too slow
frame_counter = 0
fps = 50.0
num_frames = int(np.ceil(wav_duration * fps))
print(num_frames)

last_time = datetime.datetime.now()

found_times_s_scores = [
    (w[1] / 1000, w[2]) for w in all_found_words_w_scores if w[0] == target_word
]

last_ix_transcriptions = 0
last_ix_found = 0
score_ix = 0
enable_raw_scores=False

for frame_ix in range(num_frames):
    if frame_ix % (num_frames // 100) == 0:
        now = datetime.datetime.now()
        print(frame_ix, "/", num_frames, now - last_time)
        last_time = now

    frame_time_s = frame_ix * 1 / fps

    words_times, last_ix_transcriptions = get_words(
        transcription, last_ix_transcriptions, frame_time_s
    )

    detections, last_ix_found = get_detections(
        found_times_s_scores, words_times, last_ix_found
    )

    if frame_time_s >= 1 and enable_raw_scores:
        # scores take at least one second to begin accumulating
        scores = raw_inferences[score_ix]
        confidences = dict(silence=scores[0], unknown=scores[1], merchant=scores[2])
        score_ix += 1
    else:
        confidences = dict()

    frame = make_frame(words_times, detections, confidences, threshold)

    frame_name = str(frames_dir / f"{frame_ix:05d}.jpg")
    cv.imwrite(frame_name, frame)


#%%
# in frames dir (https://stackoverflow.com/questions/13041061/mix-audio-video-of-different-lengths-with-ffmpeg)
# ffmpeg -framerate 50 -pattern_type glob -i '*.jpg' -i ../stream.wav -shortest -c:v libx264 -pix_fmt yuv420p ../out.mp4


#%%
# fmt: off
n_inferences = 60888
data_samples = 19499904
#  |[---]    |
#  |..[---]  |
#  |....[---]|
# total audio length - (the last window duration (1 second)) =
#   n previous strides [n_inferences] 
print(data_samples / 16000 == wav_duration)
stride_duration_s = 20/1000
print(int(np.ceil((wav_duration - 1) / stride_duration_s)) == n_inferences)

# calculate fps from total number of inferences
time_under_inferences_s = wav_duration - 1 # skip first second (buffer for first inference)
second_per_inference = time_under_inferences_s / n_inferences
fps = 1 / second_per_inference
print("estimated fps", fps)
print("true fps == samples per second / clip stride samples", 16000/320)
# 320 == number of samples in one stride
# 20ms/stride * ( 1s / 1000ms) -> seconds per stride
# seconds per stride * (16000 samples / second) ->  samples per stride
# fmt: on

#%%

