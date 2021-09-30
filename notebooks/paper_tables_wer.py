# %%
import pandas as pd
import numpy as np
import csv

from pathlib import Path
import os
import glob

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("white")
sns.set_palette("bright")
#%matplotlib inline

# %%
# wlist = os.listdir()

ds = sorted(
    os.listdir(Path.home() / "tinyspeech_harvard/distance_sorting/closest_farthest"),
    key=lambda x: len(x),
)
for d in ds:
    print(d)

# %%
wlist = """may
did
soon
shirt
style
taken
stood
watch
happy
entire
engine
nature
you've
reading
village
outside
strange
current
musical
followed
learning
provided
difficult
political
performance
"""
CLOSEST_FARTHEST_DIR = (
    Path.home() / "tinyspeech_harvard/distance_sorting/closest_farthest/"
)
source = (
    Path.home() / "tinyspeech_harvard/distance_sorting/cv7_extractions/listening_data"
)

n_ext = {}
for w in wlist.splitlines():
    wavs = glob.glob(str(source / w / "*.wav"))
    print(w, len(wavs))
    n_ext[w] = len(wavs)
# %%
words = [
    ["may", 2, 16],
    ["did", 16, 16],
    ["soon", 3, 17],
    ["shirt", 0, 27],
    ["style", 0, 6],
    ["taken", 0, 6],
    ["stood", 0, 11],
    ["watch", 2, 14],
    ["happy", 1, 4],
    ["entire", 0, 6],
    ["engine", 0, 9],
    ["nature", 0, 6],
    ["you've", 0, 13],
    ["reading", 4, 29],
    ["village", 0, 17],
    ["outside", 0, 1],
    ["strange", 0, 9],
    ["current", 5, 2],
    ["musical", 0, 8],
    ["followed", 0, 5],
    ["learning", 1, 9],
    ["provided", 0, 2],
    ["difficult", 0, 4],
    ["political", 0, 4],
    ["performance", 3, 6],
]


def p(w):
    return [w[0], n_ext[w[0]], f"{w[1]/50*100:0.0f}%", f"{w[2]/50*100:0.0f}%"]


table_words = [p(w) for w in words]
print(table_words)

df = pd.DataFrame(data=table_words, columns=["word", "# clips", "Near WER", "Far WER"])
# df.style.hide_index().set_precision(2)
df.style.hide_index()
# %%
print(df.to_latex(index=False))

# %%
fig, ax = plt.subplots(dpi=150)
x = range(len(words))
ax.plot(x, [w[1] / 50 * 100 for w in words], label="Nearest 50")
ax.plot(x, [w[2] / 50 * 100 for w in words], label="Farthest 50")
ax.set_xticks(x)
ax.set_xticklabels([w[0] for w in words], rotation=70)
ax.set_xlabel("word")
ax.set_ylabel("WER (%)")
ax.legend(loc="upper right")

# %%
avg_closest = []
avg_farthest = []
cf = Path.home() / "tinyspeech_harvard/distance_sorting/closest_farthest"
for d in os.listdir(cf):
    if not os.path.isdir(cf / d):
        continue
    closest = cf / d / "closest" / f"{d}_closest_50_input.csv"
    with open(closest, "r") as fh:
        c = csv.reader(fh)
        for row in c:
            dist = float(row[1])
            avg_closest.append(dist)
    farthest = cf / d / "farthest" / f"{d}_farthest_50_input.csv"
    with open(farthest, "r") as fh:
        c = csv.reader(fh)
        for row in c:
            dist = float(row[1])
            avg_farthest.append(dist)
print(np.mean(avg_closest), np.std(avg_closest))
print(np.mean(avg_farthest), np.std(avg_farthest))
# %%

# %%
# load from csv

# f = "closest_farthest_es.csv"
f = "closest_farthest_de.csv"
df = pd.read_csv(f)
word = "word"
tne = "total_num_extracted"
ngc = "num_good_closest_50"
nbc = "num_bad_closest_50"
ngf = "num_good_farthest_50"
nbf = "num_bad_farthest_50"

# sanity checks
assert np.allclose((df[ngc] + df[nbc]).values, 50)
assert np.allclose((df[ngf] + df[nbf]).values, 50)

df = df[[word, tne, nbc, nbf]]

print("Near WER mean, std", np.mean(df[nbc].values / 50 * 100), np.std(df[nbc].values / 50 * 100))
print("Far WER mean, st", np.mean(df[nbf].values / 50 * 100), np.std(df[nbf].values / 50 * 100))

# ES:
# Near WER mean 4.64
# Far WER mean 21.36
# DE
# Near WER mean 1.04, 5.14
# Far WER mean 32.32, 19.21


def to_pct(num_bad):
    return f"{num_bad/50 * 100:0.0f}%"


df[nbc] = df[nbc].apply(to_pct)
df[nbf] = df[nbf].apply(to_pct)


df = pd.DataFrame(data=df.values, columns=["Word", "# Clips", "Near WER", "Far WER"])
# df.style.hide_index().set_precision(2)
df.style.hide_index()
# %%
print(df.to_latex(index=False, column_format="lrrr"))

# %%

# %%
basedir = Path.home() / "tinyspeech_harvard/analysis_mswc"
posts = sorted(list((basedir / "POST_data").glob("*.csv")))
zscs = sorted(list((basedir / "data_ZSC").glob("*.csv")))
print(posts, zscs)

posts = list(
    zip(
        posts,
        ["German", "Greek", "Russian", "Turkish", "Vietnamese", "Chinese (zh-CN)"],
    )
)
zscs = list(
    zip(
        zscs, ["German", "Greek", "Russian", "Turkish", "Vietnamese", "Chinese (zh-CN)"]
    )
)

# %%

# %%
ix = 5
df = pd.read_csv(posts[ix][0])
lang = posts[ix][1]
mapping = {
    df.columns[0]: "Category",
    df.columns[1]: "#K",
    df.columns[2]: lang,
    df.columns[3]: "#C",
}
df = df.rename(columns=mapping)
df = df[[df.columns[i] for i in [0, 1, 3, 2]]]
df

# %%
# print(df.to_latex(index=False, column_format="cccc", caption="todo", label="todo"))
latex = df.to_latex(index=False, column_format="|c|c|c|c|")


def wrap_otherlang(text):
    ########### CHANGE THIS            VVVVVV
    start = r"""\begin{otherlanguage*}{vietnamese}"""
    end = r"""\end{otherlanguage*}"""
    return f"{start} {text} {end}"


def wrap_cn(text):
    # https://tex.stackexchange.com/a/165323
    start = r"""\begin{CJK}{UTF8}{gbsn}"""
    end = r"""\end{CJK}"""
    return f"{start} {text} {end}"


# print(wrap_otherlang("foo"))


def add_wraplang(latex):
    lines = latex.splitlines()
    preamble = lines[:4]
    end = lines[-2:]
    contents = lines[4:-2]
    # print("\n".join(preamble + end))

    new_contents = []
    for line in contents:
        ampersands = [i for i, c in enumerate(line) if c == "&"]
        last_ampersand = ampersands[-1]
        nextline = line.find("\\\\")

        examples = line[last_ampersand + 1 : nextline]

        # CHANGE THIS VVVVVVVVV
        # examples = " " + wrap_otherlang(examples) + " "
        examples = " " + wrap_cn(examples) + " "

        newline = line[: last_ampersand + 1] + examples + line[nextline:]

        new_contents.append(newline)
    print("\n".join(preamble + new_contents + end))


add_wraplang(latex)

# %%
