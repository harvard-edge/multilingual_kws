#%%

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np

from pathlib import Path
import collections
from collections import Counter

from tqdm import tqdm

# import plotly.express as px

base = Path("/mnt/disks/std750")

# data = pd.read_csv(base / "data" / "csvs" / "new4.csv")
main_data = pd.read_csv(base / "data" / "csvs" / "final-v2.csv")
main_data = main_data[main_data["counts"] >= 5]

# %%
def stacked_audio_vs_wl(main_data):
	languages = main_data['language'].unique()
	main_data['wl'] = main_data['word'].str.len()
	wordlengths_counts = dict(main_data[['counts','wl']].groupby('wl').sum().reset_index().values.tolist())
	language_fractions = {}
	for l in tqdm(languages):
		language_fractions[l] = dict(main_data[main_data['language']==l].groupby('wl')['counts'].sum().reset_index().values)


	for l in tqdm(language_fractions):
		for j in wordlengths_counts:
			if j not in language_fractions[l]:
				language_fractions[l][j] = 0

	language_fractions_df = pd.DataFrame(language_fractions)
	language_fractions_df = language_fractions_df.rename(iso_code_to_name, axis=1)
	wl15 = [3,4,5,6,7,8,9,10,11,12,13,14,15]

	language_fractions_df_T = language_fractions_df.transpose()[wl15]

	language_fractions_df_T.nsmallest(50,3).nlargest(10, 3).transpose().plot(use_index=True,kind='bar',stacked=True, figsize=(10,10), ylabel="Number of Audio Clips", xlabel='Word Length', title='Number of Audio Clips by Word Length (1 to 15) for top (1 to 10) languages', colormap='tab20')#, logy=True)
	# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

	# plt.savefig('plots/stacked/audio_clips_by_word_length_vs_word_length_top1-10.png')
	# plt.savefig('plots/stacked/audio_clips_by_word_length_vs_word_length_top1-10.pdf')

	language_fractions_df_T.nsmallest(40,3).nlargest(20, 3).transpose().plot(use_index=True,kind='bar',stacked=True, figsize=(10,10), ylabel="Number of Audio Clips", xlabel='Word Length', title='Number of Audio Clips by Word Length (1 to 15) for top (11 to 20) languages', colormap='tab20')
	# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

	# plt.savefig('plots/stacked/audio_clips_by_word_length_vs_word_length_top11-20.png')
	# plt.savefig('plots/stacked/audio_clips_by_word_length_vs_word_length_top11-20.pdf')

	language_fractions_df_T.nsmallest(20,3).nlargest(10, 3).transpose().plot(use_index=True,kind='bar',stacked=True, figsize=(10,10), ylabel="Number of Audio Clips", xlabel='Word Length', title='Number of Audio Clips by Word Length (1 to 15) for top (21 to 30) languages', colormap='tab20')
	# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

	# plt.savefig('plots/stacked/audio_clips_by_word_length_vs_word_length_top21-30.png')
	# plt.savefig('plots/stacked/audio_clips_by_word_length_vs_word_length_top21-30.pdf')

	# language_fractions_df_T.nsmallest(20,3).nlargest(10, 3).transpose().plot(use_index=True,kind='bar',stacked=True, figsize=(10,10), ylabel="Number of Audio Clips", xlabel='Word Length', title='Number of Audio Clips by Word Length (1 to 15) for top (31 to 40) languages', logy=True)
	# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

	# plt.savefig('plots/stacked/audio_clips_by_word_length_vs_word_length_top31-40.png')
	# plt.savefig('plots/stacked/audio_clips_by_word_length_vs_word_length_top31-40.pdf')

	language_fractions_df_T.nsmallest(10,3).nlargest(10, 3).transpose().plot(use_index=True,kind='bar',stacked=True, figsize=(10,10), ylabel="Number of Audio Clips", xlabel='Word Length', title='Number of Audio Clips by Word Length (1 to 15) for top (41 to 50) languages', logy=True)
	# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

	# plt.savefig('plots/stacked/audio_clips_by_word_length_vs_word_length_top51-50.png')
	# plt.savefig('plots/stacked/audio_clips_by_word_length_vs_word_length_top51-50.pdf')


#%%
iso_code_to_name = {'ab':'Abkhaz',
	'ar':'Arabic',
	'as':'Assamese',
	'br':'Breton',
	'cnh':'Hakha Chin',
	'ca':'Catalan',
	'cs':'Czech',
	'cv':'Chuvash',
	'cy':'Welsh',
	'de':'German',
	'dv':'Divehi',
	'el':'Greek',
	'en':'English',
	'es':'Spanish',
	'eo':'Esperanto',
	'eu':'Basque',
	'et':'Estonian',
	'fa':'Persian',
	'fr':'French',
	'fy-NL':'Frisian',
	'ga-IE':'Irish',
	'ia':'Interlingua',
	'id':'Indonesian',
	'it':'Italian',
	'ja':'Japanese',
	'ka':'Georgian',
	'ky':'Kyrgyz',
	'lv':'Latvian',
	'mn':'Mongolian',
	'mt':'Maltese',
	'nl':'Dutch',
	'or':'Oriya',
	'pa-IN':'Punjabi',
	'pl':'Polish',
	'pt':'Portuguese',
	'rm-sursilv':'Sursilvan',
	'rm-vallader':'Vallader',
	'ro':'Romanian',
	'ru':'Russian',
	'rw':'Kinyarwanda',
	'sah':'Sakha',
	'sl':'Slovenian',
	'sv-SE':'Swedish',
	'ta':'Tamil',
	'tr':'Turkish',
	'tt':'Tatar',
	'uk':'Ukrainian',
	'vi':'Vietnamese',
	'zh-CN':'Chinese',
	'zh-TW':'Chinese'}

low = ["mt","br","rm-sursilv","sl","sah","lv","cv","ga-IE","ka","cnh","ha","rm-vallader","vi","as","gn","ab","or"]
medium = ["eu","nl","pt","tt","cs","uk","et","tr","mn","ky","ar","fy-NL","sv-SE","id","el","ro","ia","sk","zh-CN","dv"]
high = ["de","en","fr","ca","rw",'es',"ru","it","pl","fa","eo","cy","ta"]

iso_code_to_name['gn'] = "Gaurian"
iso_code_to_name['ha'] = "Hausa"
iso_code_to_name['sk'] = "Slovak"


# stacked_audio_vs_wl(main_data)

def stacked_audio_vs_wl_new(main_data):
	languages = main_data['language'].unique()
	main_data['wl'] = main_data['word'].str.len()
	wordlengths_counts = dict(main_data[['counts','wl']].groupby('wl').sum().reset_index().values.tolist())
	language_fractions = {}
	for l in tqdm(languages):
		language_fractions[l] = dict(main_data[main_data['language']==l].groupby('wl')['counts'].sum().reset_index().values)


	for l in tqdm(language_fractions):
		for j in wordlengths_counts:
			if j not in language_fractions[l]:
				language_fractions[l][j] = 0

	language_fractions_df = pd.DataFrame(language_fractions)
	language_fractions_df = language_fractions_df.rename(iso_code_to_name, axis=1)
	wl15 = [3,4,5,6,7,8,9,10]

	high_lang = [iso_code_to_name[i] for i in high]
	low_lang = [iso_code_to_name[i] for i in low]
	medium_lang = [iso_code_to_name[i] for i in medium]

	language_fractions_df_T = language_fractions_df.transpose()[wl15]

	language_fractions_df_T.transpose()[low_lang].plot(use_index=True,kind='bar',stacked=True, figsize=(10,10), ylabel="Number of Audio Clips", xlabel='Word Length', colormap='tab20')
	language_fractions_df_T.transpose()[medium_lang].plot(use_index=True,kind='bar',stacked=True, figsize=(10,10), ylabel="Number of Audio Clips", xlabel='Word Length', colormap='tab20')
	language_fractions_df_T.transpose()[high_lang].plot(use_index=True,kind='bar',stacked=True, figsize=(10,10), ylabel="Number of Audio Clips", xlabel='Word Length', colormap='tab20')


stacked_audio_vs_wl_new(main_data)
# %%