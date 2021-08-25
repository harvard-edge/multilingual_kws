#%%
import os
import glob
import shutil
from pathlib import Path
import subprocess
import sox
import multiprocessing

# %%
train_unknown_f = Path.cwd() / "msc_train_other_files.txt"
with open(train_unknown_f, "r") as fh:
    train_unknown = fh.read().splitlines()
test_unknown_f = Path.cwd() / "msc_test_other_files.txt"
with open(test_unknown_f, "r") as fh:
    test_unknown = fh.read().splitlines()
print(len(train_unknown), len(test_unknown))

# %%
overlap = len(set(train_unknown).intersection(test_unknown))
assert overlap == 0
# %%
# check
bad = []
extra_bad = 0
for f in train_unknown + test_unknown:
    cmd = f"soxi {f}"
    try:
        res = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        bad.append(f)
        extra_bad += 1
        continue
    out = res.decode("utf8")
    if not "75 CDDA sectors" in out:
        # print(f"fail on {f}, {out}")
        bad.append(f)
    if not "48000" in out:
        # print(f"fail on {f}, {out}")
        bad.append(f)
print(len(bad))
print(extra_bad)

# %%
bset = set(bad)
ok_train_unknown = [f for f in train_unknown if f not in bset]
ok_test_unknown = [f for f in test_unknown if f not in bset]
print(len(ok_train_unknown), len(ok_test_unknown))

# %%
ok_train_unknown_f = Path.cwd() / "ok_msc_train_other_files.txt"
with open(ok_train_unknown_f, "w") as fh:
    fh.write("\n".join(ok_train_unknown))
ok_test_unknown_f = Path.cwd() / "ok_msc_test_other_files.txt"
with open(ok_test_unknown_f, "w") as fh:
    fh.write("\n".join(ok_test_unknown))

# %%
train_dest = Path("/mnt/disks/std750/mark/msc_other_files/train/")
test_dest = Path("/mnt/disks/std750/mark/msc_other_files/test/")

def convert_16k_wav(file_destdir):
    f = file_destdir[0]
    destdir = file_destdir[1]

    word = Path(f).parts[-2]

    os.makedirs(destdir / word, exist_ok=True)
    dest = destdir / word / Path(f).name
    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s
    transformer.build(f, str(dest))
    return f

for files, dest in zip([ok_train_unknown, ok_test_unknown], [train_dest, test_dest]):
    files_dests = [(f, dest) for f in files]
    print(files_dests)

    with multiprocessing.Pool() as p:
        done = []
        for f in p.imap_unordered(convert_16k_wav, files_dests):
            done.append(f)
    print(len(done))

    
# %%
f = ok_test_unknown[0]
Path(f).parts[-2]
# %%
basedir = Path.cwd()
original = basedir / "orig_gsc_msc"
target = basedir / "gsc_msc"

def convert_16k_wav_dir(word):
    os.makedirs(target / word, exist_ok=True)
    for wav in glob.glob(str(original / word / "*.wav")):
        dest = target / word / Path(wav).name
        transformer = sox.Transformer()
        transformer.convert(samplerate=16000)  # from 48K mp3s
        transformer.build(wav, str(dest))
    return word
# below is per directory, way slower than above which works per-file 
with multiprocessing.Pool() as p:
    for i in p.imap_unordered(convert_16k_wav_dir, os.listdir(original)):
        print(i)
print("done")
# %%
