#%%
from pathlib import Path
import subprocess, os, multiprocessing

data = Path("/mnt/disks/std3/final/audio/")
files = os.listdir(data)

paths = []

for f in files:
	if f in ["en.tar.gz", "fr.tar.gz", "de.tar.gz", "ca.tar.gz", "rw.tar.gz"]:
		paths.append(data / f)

print(len(paths))
# exit(0)
#%%
def upload(path):
	subprocess.call(['rclone', 'copy', str(path), 'msc3:MSC_Sharad/'])

pool = multiprocessing.Pool()
for i, result in enumerate(pool.imap_unordered(upload, paths), start=1):
	pass