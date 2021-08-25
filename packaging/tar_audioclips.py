#%%
import os
from pathlib import Path

audio_clips = Path("/mnt/disks/std3/compressed_cleaned3/generated/common_voice/frequent_words")
languages = os.listdir(audio_clips)

#%%
final_audio_clips = Path("/mnt/disks/std3/final/audio2/")

os.chdir(final_audio_clips)

# %%
from tqdm import tqdm

for l in tqdm(languages):
	old_path = audio_clips / l
	os.system(f"tar -cf {l}.tar {old_path}")
