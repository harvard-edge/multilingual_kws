import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np
import argparse
from argparse import ArgumentParser

import pathlib
from pathlib import Path

import collections
from collections import Counter
from tqdm import tqdm

def main():
    """
    Create argument parser
    """
    parser = ArgumentParser()
    # An argument of type string that takes directory name as input
    parser.add_argument("--output_file", type=str, help="output file")
    # An argument of type filename that takes as input string to tell which file to read
    parser.add_argument("--input_file", type=str, help="input file")
    # argument for output directory
    parser.add_argument("--output_dir", type=str, help="output directory")
    # argument for input directory
    parser.add_argument("--input_dir", type=str, help="input directory")
    args = parser.parse_args()
    return args

def get_paths(args):
    input_file = Path(args.input_dir) / args.input_file
    output_file = Path(args.output_dir) / args.output_file
    print(input_file, output_file)
    return input_file, output_file

if __name__ == "__main__":
    args = main()
    # Read the file
    input_file, output_file = get_paths(args)

    stats = pickle.load(open(input_file,'rb'))

    data =[]
    for l in tqdm(stats):
        counts = stats[l]
        counts = Counter(counts).most_common(50000)
        datat = [(i, counts[i][0], counts[i][1], l) for i in range(len(counts))]
        data.extend(datat)

    df = pd.DataFrame(data, columns=['index','word','counts','language'])

    # Save file
    df.to_csv(output_file, index=False)


