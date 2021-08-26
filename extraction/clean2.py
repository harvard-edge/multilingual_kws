#%%
import shutil, multiprocessing, pickle
from pathlib import Path

data = Path("/mnt/disks/std3/compressed/generated/common_voice/frequent_words/")
new_data = Path("/mnt/disks/std3/compressed_cleaned2/generated/common_voice/frequent_words/")

map = pickle.load(open("cleaned_LT.p",'rb'))

def create_paths(map):
	old_paths, new_paths = [], []
	for l in tqdm(map):
		if l in ['ja','pa-IN']:
			continue
		lpath = data / l / "clips"

		for w in map[l]:
			old_paths.append(lpath / w)
			new_paths.append(new_data / l / "clips" / map[l][w])

	return old_paths, new_paths

from tqdm import tqdm

old_paths, new_paths = create_paths(map)
#%%
print(f"Total paths {len(old_paths)}")
def copy(oword, nword):
	shutil.copytree(oword, nword, dirs_exist_ok=True)

pool = multiprocessing.Pool()
with tqdm(total=len(new_paths)) as pbar:
	for i, result in enumerate(pool.starmap(copy, zip(old_paths, new_paths)), start=1):
		pass
# %%
