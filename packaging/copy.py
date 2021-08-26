import shutil, multiprocessing, pickle, os
from pathlib import Path

from tqdm import tqdm

old_data = Path("/mnt/disks/std3/compressed_cleaned3/generated/common_voice/frequent_words/")
new_data = Path("/mnt/disks/packaging/data/generated/common_voice/frequent_words/")

languages = os.listdir(old_data)
old_paths, new_paths = [], []

for l in tqdm(languages):
	words = os.listdir(old_data / l / "clips")
	for w in words:
		old_paths.append(old_data / l / "clips" / w)
		new_paths.append(new_data / l / "clips" / w)

def copy(oword, nword):
	shutil.copytree(oword, nword)

pool = multiprocessing.Pool()
for i, result in enumerate(pool.starmap(copy, zip(old_paths, new_paths)), start=1):
	pass