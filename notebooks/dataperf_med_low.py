# %%
from pathlib import Path
import isocodes
import subprocess
from datasets import load_dataset
import tqdm
import collections

# %%
# fmt: off
lr_languages = """Arabic (0.1G, 7.6h), Assamese (0.9M, 0.1h), Breton (69M, 5.6h), Chuvash (28M, 2.1h), Chinese (zh-CN) (42M, 3.1h),
Dhivehi (0.7M, 0.04h), Frisian (0.1G, 9.6h), Georgian (20M, 1.4h), Guarani (0.7M, 1.3h), Greek (84M, 6.7h),
Hakha Chin (26M, 0.1h), Hausa (90M, 1.0h), Interlingua (58M, 4.0h), Irish (38M, 3.2h), Latvian (51M, 4.2h),
Lithuanian (21M, 0.46h), Maltese (88M, 7.3h), Oriya (0.7M, 0.1h), Romanian (59M, 4.5h),
Sakha (42M, 3.3h), Slovenian (43M, 3.0h), Slovak (31M, 1.9h), Sursilvan (61M, 4.8h),
Tamil (8.8M, 0.6h), Vallader (14M, 1.2h), Vietnamese (1.2M, 0.1h)"""

mr_languages = """Czech (0.3G, 24h), Dutch (0.8G, 70h), Estonian (0.2G, 19h), Esparanto (1.3G, 77h),
Indonesian (0.1G, 11h), Kyrgyz (0.1G, 12h), Mongolian (0.1G, 12h), Portuguese (0.7G, 58h),
Swedish (0.1G, 12h), Tatar (4G, 30h), Turkish (1.3G, 29h), Ukranian (0.2G, 18h)"""

clean = lambda languages: [x.split(" (")[0].strip() for x in languages.replace("\n", "").split("),")]

lr_languages_clean = clean(lr_languages)
mr_languages_clean = clean(mr_languages)
# fmt: on

lr_languages_clean = [isocodes.languages[l] for l in lr_languages_clean]
mr_languages_clean = [isocodes.languages[l] for l in mr_languages_clean]

# restrict to simple isocodes
lr_languages_clean = [l for l in lr_languages_clean if len(l) == 2]
mr_languages_clean = [l for l in mr_languages_clean if len(l) == 2]
print(lr_languages_clean)
print(mr_languages_clean)

# ['ar', 'as', 'br', 'cv', 'dv', 'ka', 'gn', 'el', 'ha', 'ia', 'lv', 'lt', 'mt', 'or', 'ro', 'sl', 'sk', 'ta', 'vi']
# ['cs', 'nl', 'et', 'eo', 'id', 'ky', 'mn', 'pt', 'tt', 'tr', 'uk']



# ['ar', 'as', 'br', 'cv', 'zh-CN', 'dv', 'fy-NL', 'ka', 'gn', 'el', 'cnh', 'ha', 'ia', 'ga-IE', 'lv', 'lt', 'mt', 'or', 'ro', 'sah', 'sl', 'sk', 'rm-sursilv', 'ta', 'rm-vallader', 'vi']
# ['cs', 'nl', 'et', 'eo', 'id', 'ky', 'mn', 'pt', 'sv-SE', 'tt', 'tr', 'uk']
# %%

# ds = load_dataset("MLCommons/ml_spoken_words", "ro_wav", cache_dir="/media/mark/sol/dpml/")
# %%
# for d in ds["train"]:
#     print(d)
#     break

# %%

# for l in tqdm.tqdm(['nl', 'pt', 'id'] + lr_languages_clean):
#     l_fmt = f"{l}_wav"
#     print(l_fmt)
#     ds = load_dataset("MLCommons/ml_spoken_words", l_fmt, cache_dir="/media/mark/sol/dpml/")


# %%
# {'file': '/media/mark/sol/dpml/downloads/extracted/35026456632e4d61cdccb4124f24091c480d25d9a9e6c550e6fb91ef8a76c4c3/abordare_common_voice_ro_20352187.wav', 'is_valid': True, 'language': 0, 'speaker_id': 'b857f50db91f0dde263d96733dd644a702e130919d233ce11a67a214357c30d40d692753ff8eb61756e288ea78fe2300678a4a1213dfa44d98ac9feeadb935bd', 'gender': 0, 'keyword': 'abordare', 'audio': {'path': '/media/mark/sol/dpml/downloads/extracted/35026456632e4d61cdccb4124f24091c480d25d9a9e6c550e6fb91ef8a76c4c3/abordare_common_voice_ro_20352187.wav', 'array': array([0., 0., 0., ..., 0., 0., 0.], dtype=float32), 'sampling_rate': 16000}}

# %%
# find common words in indonesian
c = collections.Counter()
ds = load_dataset("MLCommons/ml_spoken_words", "id_wav", cache_dir="/media/mark/sol/dpml/")
for d in ds["validation"]:
    c[d['keyword']] += 1
c.most_common(100)
# %%
# val count
for kw in ["karena", "sangat", "bahasa", "belajar", "kemarin"]:
    print(kw, c[kw])
# %%
# train counts:
# karena: because, 181
# sangat: very, 159
# bahasa: language, 135 
# belajar: study, 107
# kemarin: yesterday, 103
# val counts:
# karena 25
# sangat 22
# bahasa 19
# belajar 14
# kemarin 13

# %%
# find common words in portuguese
c = collections.Counter()
ds = load_dataset("MLCommons/ml_spoken_words", "pt_wav", cache_dir="/media/mark/sol/dpml/")
for d in ds["train"]:
    c[d['keyword']] += 1
c.most_common(100)

# %%
# train count
for kw in ["pessoas", "grupo", "camisa", "tempo", "andando"]:
    print(kw, c[kw])
# pessoas 1042
# grupo 383
# camisa 354
# tempo 375
# andando 320

# %%
# val counts
# pessoas: people, 133
# grupo: group, 48
# camisa: shirt, 47
# tempo: time, 47
# andando: walking, 41
