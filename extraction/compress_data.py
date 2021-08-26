import os, multiprocessing
import sox

from pathlib import Path

from tqdm import tqdm

def collect_language_path():
	path, newpath = "/mnt/disks/std3/data/generated/common_voice/frequent_words/{0}/clips/", "/mnt/disks/std3/compressed/generated/common_voice/frequent_words/{0}/clips/", 
	# language_directory = path.format(lang)
	old_clip_paths, new_clip_paths = [], []
	# languages = "ca  cv  de  el  en  fa  lt  pt  ru  rw  ta  tr  tt  uk".split('  ') 
	# languages = ['ab', 'ar', 'as', 'cnh', 'cs', 'cy', 'dv', 'el', 'eo', 'es', 'et', 'eu', 'fi', 'fr', 'fy-NL', 'ga-IE', 'hi', 'ia', 'id', 'it', 'ja', 'ka', 'ky', 'lg', 'lv', 'mn', 'mt', 'nl', 'or', 'pa-IN', 'pl', 'rm-sursilv', 'rm-vallader', 'ro', 'sah', 'sl', 'sv-SE', 'vi']
	# languages = ['zh-CN','gn','ha', 'sk']
	languages = ['lt']
	# languages = "br  ca  cv  de  el  en  fa  lt  pt  ru  rw  ta  tr  tt  uk".split('  ') 

	# languages = ["br"]
	# path, newpath = "/mnt/disks/std3/data/generated/common_voice/frequent_words/{0}/clips/", "/mnt/disks/std3/compressed/generated/common_voice/frequent_words/{0}/clips/", 
	for lang in tqdm(languages):
		language_directory = path.format(lang)
		for word in os.listdir(language_directory):
			word_directory = os.path.join(language_directory, word)
			new_path_word_directory = os.path.join(newpath.format(lang), word)

			old_clip_paths.append(word_directory)
			new_clip_paths.append(new_path_word_directory)

			os.makedirs(new_path_word_directory, exist_ok=True)
			# for clip in os.listdir(word_directory):
			# 	newp = os.path.join(new_path_word_directory, clip.replace('.wav', '.mp3'))
			# 	if os.path.exists(newp):
			# 		continue
			# 	paths.append(os.path.join(word_directory, clip))
			# 	newpaths.append(newp)

	return old_clip_paths, new_clip_paths

def collect_paths():
	"""
	Collects the paths to all the files in the data directory.
	"""
	paths, newpaths = [], []
	languages = "ca  cv  de  el  en  fa  lt  pt  ru  rw  ta  tr  tt  uk".split('  ') 
	# languages = "br  ca  cv  de  el  en  fa  lt  pt  ru  rw  ta  tr  tt  uk".split('  ') 

	# languages = ["br"]
	path, newpath = "/mnt/disks/std3/data/generated/common_voice/frequent_words/{0}/clips/", "/mnt/disks/std3/compressed/generated/common_voice/frequent_words/{0}/clips/", 
	for lang in tqdm(languages):
		language_directory = path.format(lang)
		for word in os.listdir(language_directory):
			word_directory = os.path.join(language_directory, word)
			new_path_word_directory = os.path.join(newpath.format(lang), word)

			# os.makedirs(new_path_word_directory, exist_ok=True)
			for clip in os.listdir(word_directory):
				newp = os.path.join(new_path_word_directory, clip.replace('.wav', '.mp3'))
				if os.path.exists(newp):
					continue
				paths.append(os.path.join(word_directory, clip))
				newpaths.append(newp)
	return paths, newpaths

def compress(oldpath, newpath):
# def compress():
	# oldpath, newpath = "/mnt/disks/std3/data/generated/common_voice/frequent_words/en/clips/above/common_voice_en_18923823.wav", "test.mp3"
	# try:
	transformer = sox.Transformer()
	transformer.convert(samplerate=48000)  # from 48K mp3s
	transformer.build(str(oldpath), str(newpath))
	# transformer.fade(fade_in_len=0.025, fade_out_len=0.025)
	# transformer.pad(start_duration=pad_amt_s, end_duration=pad_amt_s)
	# except Exception as e:
	# 	print(e)

	return True

def main():
	uncompressed_audios, newpaths = collect_paths()
	print(f"Total files {len(newpaths)}")
	pool = multiprocessing.Pool()
	with tqdm(total=len(newpaths)) as pbar:
		for i, result in enumerate(pool.starmap(compress, zip(uncompressed_audios, newpaths)), start=1):
			if i%50000 == 0:
				print(f"{i} files processed")


def remove_paths(old, new):
	newoldpaths = []
	processed_paths = []
	# for i, word_p in os.listdir(old):
	# assert len(old) == len(new), f"Length of old {len(old)} does not match length of new {len(new)}"
	for clip in os.listdir(old):
		newp = os.path.join(new, clip.replace('.wav', '.mp3'))
			# newp = os.path.join(new_path_word_directory, clip.replace('.wav', '.mp3'))
		if os.path.exists(newp):
			continue
		newoldpaths.append(newp)
		processed_paths.append(os.path.join(old, clip))
	return processed_paths, newoldpaths


def main2():
	paths, newpaths = collect_language_path()
	print(f"Total paths to process {len(newpaths)}")
	pool = multiprocessing.Pool()
	oldpaths, newprocessed_paths = [], []
	# with tqdm(total=len(newpaths)) as pbar:
	for i, (r1, r2) in enumerate(pool.starmap(remove_paths, zip(paths, newpaths)), start=1):
		oldpaths.extend(r1)
		newprocessed_paths.extend(r2)
		# if i%5000 == 0:
		# 	print(f"{i} paths processed")

	# pool = multiprocessing.Pool()
	print(f"Total files to process {len(newprocessed_paths)}")
	for i, result in enumerate(pool.starmap(compress, zip(oldpaths, newprocessed_paths)), start=1):
		# if i%50000 == 0:
		# 	print(f"{i} files processed")
		hello = 1
	

	

if __name__ == '__main__':
	main2()
	# compress()
	