#%%
import os
from pathlib import Path
from tqdm import tqdm

old_data = Path("/mnt/disks/std3/compressed_cleaned3/generated/common_voice/frequent_words/")
# new_data = Path("/mnt/disks/std3/compressed_cleaned3/generated/common_voice/frequent_words/")

languages = os.listdir(old_data)

words = {}
for l in tqdm(languages):
	words[l] = set(os.listdir(old_data / l / "clips"))
# %%

lt3 = 0
lt3s = []
old_paths, new_paths = [], []
for l in tqdm(languages):
	for w in list(words[l]):
		if len(w) < 3 and l!='zh-CN':
			lt3 += 1
			lt3s.append(f"{l}/clips/{w}")
		elif len(w) < 2 and l=='zh-CN':
			lt3 += 1
			lt3s.append(f"{l}/clips/{w}")
		else:
			# new_path = new_data / l / "clips" / w
			old_path = old_data / l / "clips" / w
			# if os.path.exists(new_path):
			# 	pass
			# else:
			# 	if old_path.is_dir():
			# 		# copy(old_path, new_path)
			# 		old_paths.append(old_data / f"{l}/clips/{w}")
			# 		new_paths.append(new_data / f"{l}/clips/{w}")
# %%
import shutil, multiprocessing, pickle
print(f"Total keywords {len(old_paths)}")
print(f"Skipping {lt3} keywords")


def copy(oword, nword):
	shutil.copytree(oword, nword)

pool = multiprocessing.Pool()
for i, result in enumerate(pool.starmap(copy, zip(old_paths, new_paths)), start=1):
	pass
# %%
