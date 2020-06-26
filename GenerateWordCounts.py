import pandas as pd
import os
import argparse
import sys
import csv
import nltk
from typing import Set, List, Dict
import functools

def clean_and_filter(keyword_set: Set[str], sentence: str) -> List[str]:
    tokens = nltk.word_tokenize(sentence)
    words = [word.lower() for word in tokens if word.isalpha()]
    return list(filter(lambda w: w in keyword_set, words))

def update_counts(counts: Dict[str, int], words: List[str]) -> None:
    for w in words:
        counts[w] += 1

def count_occurences(keyword_set, tsv):
    df = pd.read_csv(tsv, sep="\t")
    
    # there are NaNs in test.tsv['sentence']
    # https://stackoverflow.com/a/50533971
    df.sentence.dropna(inplace=True)
    
    print("Dataset:", tsv)
    print("Number of mp3s:", df.shape[0])

    counts = {k:0 for k in keyword_set}
    
    # TODO(MMAZ) inefficient
    df['keywords'] = df.sentence.apply(functools.partial(clean_and_filter, keyword_set))
    
    df.keywords.dropna(inplace=True)
    usable = df.keywords.transform(len)
    print("mp3s containing speechcommands keywords", usable[usable > 0].shape[0])
    
    _ = [update_counts(counts, tokens) for tokens in df.keywords]
    
    return counts

def create_counts_df(counts):
    counts_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    counts_df.columns = ['Keyword', 'Word Count']
    counts_df.set_index('Keyword', inplace=True)
    return counts_df

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Parses command.')

    parser.add_argument("--keywords_input", action="store", dest="keywords_input", help="Point to Keywords JSON file")
    parser.add_argument("--tsv_input", action="store", dest="tsv_input", help="Point to TSV file with mp3 files")
    parser.add_argument("--output", action="store", dest="output", help="Point to output csv location")

    options = parser.parse_args(args)

    keywords = set([k.strip() for k in open(options.keywords_input).readline().split(',')])

    selected_counts = count_occurences(keywords, options.tsv_input)

    counts_df = create_counts_df(selected_counts)
    counts_df.to_csv(options.output)

main()