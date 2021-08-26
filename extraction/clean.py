#%%
from cvutils import Alphabet
from cvutils import Validator
from pathlib import Path
from tqdm import tqdm
import os, multiprocessing, shutil

data = Path('/mnt/disks/std3/compressed/generated/common_voice/frequent_words/')

# langs = os.listdir(data)
langs = ['lt']

# %%
map, nolang = {}, set()
# langs = ['en']

#%%
# for l in tqdm(langs):
# 	map[l] = {}
# 	try:
# 		v = Validator(l)
# 	except:
# 		nolang.add(l)
# 		continue
# 	words = os.listdir(data / l / 'clips')
# 	for w in words:
# 		tw = correct_spelling(w)
# 		map[l][w] = tw
# 		out = v.validate(w)
# 		if out is not None:
# 			map[l][w] = out
# 		else:
# 			print(w)
			# map[l][w] = w

# def correct_spelling(word, alphabet):
# 	return [w for w in word if w in alphabet else '']

#%%
def correct_spelling(w, alphabet):
	nw = ''
	for i in w:
		if i in alphabet:
			nw += i
		else:
			continue
	if nw == '':
		return ''

	if nw[0] == "'":
		nw = nw[1:]
	if nw[-1] == "'":
		nw = nw[:-1]
	return nw

# langs = os.listdir(data)
map = {}
novalids, noalphas = [], []
for l in tqdm(langs):
	valid, alpha = False, False
	map[l] = {}
	try:
		v = Validator(l)
		valid = True
	except:
		novalids.append(l)
		valid = False
	try:
		alphabet = set(Alphabet(l).get_alphabet())
		alpha = True
	except:
		noalphas.append(l)
		alpha = False

	words = os.listdir(data / l / 'clips')

	if not alpha:
		map[l] = {k:k for k in words}
		continue

	for w in tqdm(words):
		nw = w
		if valid:
			nw_validated = v.validate(nw)
			if nw_validated is not None:
				nw = nw_validated
		nw = correct_spelling(nw, alphabet)
		map[l][w] = nw


cleaned = map

import pickle
with open('cleaned_LT.p','wb') as file:
	pickle.dump(cleaned, file)