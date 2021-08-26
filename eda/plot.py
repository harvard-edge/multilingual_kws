import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np

import collections
from collections import Counter

from tqdm import tqdm

import plotly.express as px

#fig =px.scatter(x=range(10), y=range(10))
#fig.write_html("path/to/file.html")

stats = pickle.load(open('frequent_words_stats_std2.p','rb'))
# wordlengths = {}

# for l in stats:
#     wordlengths[l] = len(stats[l])
# wordlengths = dict(sorted(wordlengths.items(), key=lambda item: item[1], reverse=True))

# #plt.gcf().set_size_inches(70,15)
# #plt.bar(wordlengths.keys(), wordlengths.values())
# fig, ax = plt.subplots()
# fig.set_size_inches(30,20)
# #plt.gcf().set_size_inches(70,15)
# ax.bar(wordlengths.keys(), wordlengths.values())
# ax.set_xticklabels(wordlengths.keys(), rotation=70)
# ax.set_xlabel("Languages")
# ax.set_ylabel("Number of words")
# ax.set_title("Number of Words for various languages in Common Voice")
# #ax.bar_label(p1)
# plt.savefig('numwords.png')
# plt.cla()

# wordlengths = {}

# for l in stats:
#     wordlengths[l] = sum([len(i) for i in stats[l].keys()]) / len(stats[l].keys())

# fig, ax = plt.subplots()
# fig.set_size_inches(30,20)
# #plt.gcf().set_size_inches(70,15)
# ax.bar(wordlengths.keys(), wordlengths.values())
# ax.set_xticklabels(wordlengths.keys(), rotation=70)
# ax.set_xlabel("Languages")
# ax.set_ylabel("Word Length")
# ax.set_title("Average Word Length for various languages in Common Voice")
# #ax.bar_label(p1)
# plt.savefig('wordlengths.png')
# plt.cla()

def plot_counts(counts, title, lang):
    counts = Counter(counts)
    data = [(i, counts[i][1]) for i in range(len(counts.most_common(500)))]
#    df = pd.DataFrame([data, columns=["index", "counts"]])
    #sns.barplot(x="counts", y="keyword", data=df).set_title(title)
    #length
#    print(len(np.arange(0,50)), len(counts.most_common(50)))
#    print(counts.most_common(50))
    fig = px.line(df, x='index',y='counts', color = lang)
    #plt.plot(np.arange(0,50).tolist(), [i[1] for i in counts.most_common(50)], label = lang)
    print([i[1] for i in counts.most_common(500)])
    #plt.gcf().set_size_inches(15,70)
#    plt.savefig(f"plots/words/{lang}.png")
#    plt.cla()
    return fig


data =[]
for l in tqdm(stats):
    counts = stats[l]
    counts = Counter(counts).most_common(500)
    #counts
    #print(counts)
    datat = [(i, counts[i][1], l) for i in range(len(counts))]
    data.extend(datat)

df = pd.DataFrame(data, columns=['index','counts','language'])

fig = px.line(df, x='index', y='counts', color='language')

#for l in tqdm(stats):
#    fig = plot_counts(stats[l], f"{l} frequent words", l)
    #plt.draw()
    #plt.pause(0.0001)

fig.write_html('line.html')
#plt.legend()
#plt.savefig('line.png')
#

from tqdm import tqdm

def wordlengths(main_data):
    languages = main_data['language'].unique()
    plot_saves = "plots/wordlengths/"
    for l in tqdm(languages):

        sub_df = main_data[main_data['language']==l]
        words = sub_df['word'].unique().tolist()

        wordlengths = {}
        for w in words:
            if type(w) == str:
                if len(w) in wordlengths:
                    wordlengths[len(w)] += 1
                else:
                    wordlengths[len(w)] = 1

        fig, ax = plt.subplots(figsize=(8,8))
        ax.bar(wordlengths.keys(), wordlengths.values())
        ax.set_xlabel("Word Length")
        ax.set_ylabel("Number of Keywords")
        ax.set_title(f"Number of Keywords v/s Word Length for {l}")
        save_path = plot_saves + f"{l}.png"
        fig.savefig(f"{save_path}")

def graph1():
    plot_saves = "plots/wordlengths/"
    from tqdm import tqdm
    for l in tqdm(languages):

        sub_df = main_data[main_data['language']==l]

        plotdata = sub_df[['word','counts']].groupby('counts').count().reset_index()
        # plotdata = sub_df_w_counts

        fig, ax = plt.subplots(figsize=(8,8))
        ax.bar(plotdata.counts, wordlengths.word)
        ax.set_xlabel("Word Length")
        ax.set_ylabel("Number of Keywords")
        ax.set_title(f"Number of Keywords v/s Word Length for {l}")
        save_path = plot_saves + f"{l}.png"
        fig.savefig(f"{save_path}")