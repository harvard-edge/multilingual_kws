#%%
import os
from pathlib import Path

old_alignments = Path("/mnt/disks/std750/data/common-voice-forced-alignments/")
languages = os.listdir(old_alignments)

to_remove = ['pa-IN','ja','zh-CN','zh-TW','zh-HK','README.md','LICENSE.md']
languages = [x for x in languages if x not in to_remove]

#%%
final_alignments = Path("/mnt/disks/std3/final/alignments/")

os.chdir(final_alignments)

# %%
from tqdm import tqdm

for l in tqdm(languages):
	old_path = old_alignments / l
	os.system(f"tar -cf {l}.tar.gz {old_path}")
# %%

new_alignments = Path("/mnt/disks/std3/alignments")
languages = os.listdir(new_alignments)

for l in tqdm(languages):
	new_path = new_alignments / l
	os.system(f"tar -cf {l}.tar.gz {new_path}")
# %%
