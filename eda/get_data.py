#%%

import numpy as np
import pandas as pd
import argparse
from argparse import ArgumentParser

from typing import Dict

def argparser():
	"""
	Argument parser
	"""
	parser = ArgumentParser()
	parser.add_argument("--input_file", type=str, help="input file")
	parser.add_argument("--output_file", type=str, help="output file")
	args = parser.parse_args()

	return args

def get_number_of_keywords(df: pd.DataFrame, min_num_of_extractions: int = 20) -> Dict[str, int]:
	langs = np.unique(df['language'].values)
	vals = {}
	for l in langs:
		tdf = df[df['language'] == l]
		tdf = tdf[tdf['counts']>=min_num_of_extractions]
		vals[l] = len(tdf)

	return vals

def max_and_min_num_extraction_each_language(df: pd.DataFrame):
	langs = np.unique(df['language'].values)
	min, max = {}, {}
	for l in langs:
		max[l], min[l] = (df[df['language'] == l]['counts'].max(), df[df['language'] == l]['counts'].min())

	return min, max

def get_avg_extractions(df: pd.DataFrame):
	langs = np.unique(df['language'].values)
	vals = {}
	for l in langs:
		tdf = df[df['language'] == l]
		vals[l] = tdf['counts'].mean()

	return vals
#%%
if __name__ == '__main__':
	args = argparser()
	# input_file = "../../data/csvs/new.csv"
	df = pd.read_csv(args.input_file)
		
	df = df[df.counts >= 5]
	#%%
	# vals0 = get_number_of_keywords(df, 0)
	vals5 = get_number_of_keywords(df, 5)
	vals20 = get_number_of_keywords(df, 20)
	vals50 = get_number_of_keywords(df, 50)
	vals100 = get_number_of_keywords(df, 100)
	vals200 = get_number_of_keywords(df, 200)

	# table = pd.DataFrame(dict(zip(['Language', 'Atleast 0 extractions', 'Atleast 5 extractions', 'Atleast 20 extractions', 'Atleast 50 extractions', 'Atleast 100 extractions', 'Atleast 200 extractions'], [np.unique(df['language'].values).tolist(), vals0.values(), vals5.values(), vals20.values(), vals50.values(), vals100.values(), vals200.values()])), index=vals5.keys())
	table = pd.DataFrame(dict(zip(['Language', 'Atleast 5 extractions', 'Atleast 20 extractions', 'Atleast 50 extractions', 'Atleast 100 extractions', 'Atleast 200 extractions'], [np.unique(df['language'].values).tolist(), vals5.values(), vals20.values(), vals50.values(), vals100.values(), vals200.values()])), index=vals5.keys())

	min, max = max_and_min_num_extraction_each_language(df)
	avg = get_avg_extractions(df)

	table['Minimum'] = min.values()
	table['Maximum'] = max.values()
	table['Average'] = avg.values()

	# output_file = "../../data/csvs/new2.csv"

	table.to_csv(args.output_file, index=False)


	# print(vals)
# %%
