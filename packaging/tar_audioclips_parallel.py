#%%
import os, multiprocessing
from pathlib import Path

audio_clips = Path("/mnt/disks/std3/compressed_cleaned3/generated/common_voice/frequent_words")
languages = os.listdir(audio_clips)

# languages = ["ab","ar","as","br","ca","cnh","cs","cv","cy","dv","el","eo","es","et","eu","fa","fr"]

#%%
final_audio_clips = Path("/mnt/disks/std3/final/audio/")

os.chdir(final_audio_clips)

# %%
from tqdm import tqdm
import subprocess

def tar(l, path):
	subprocess.call(["tar", "-zcf", f"{l}.tar.gz", path])#, env=dict(os.environ, **{"GZIP":"-1"}))

old_paths, ls = [], []
for l in tqdm(languages):
	ls.append(l)
	old_path = audio_clips / l
	old_paths.append(str(old_path))
	# os.system(f"tar -cf {l}.tar {old_path}")
# print(old_paths)
# exit(0)
#%%
pool = multiprocessing.Pool()
for i, result in enumerate(pool.starmap(tar, zip(ls, old_paths)), start=1):
	pass