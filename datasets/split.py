from __future__ import print_function
import torch
from torch import nn
import pandas as pd
import torchtext
from torchtext import data
import spacy
import numpy as np


def train_validate_test_split(df, train_percent=.5, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    print("m = " + str(m))
    print("end = " + str(train_end))
    print("perm[:train_end] = " + str(perm[:train_end]))
    train = df.loc[perm[:train_end]]
    validate = df.loc[perm[train_end:validate_end]]
    test = df.loc[perm[validate_end:]]
    return train, validate, test

authors_dataset = pd.read_csv("datasets_491_9097_authors.csv")
paper_authors_dataset = pd.read_csv("datasets_491_9097_paper_authors.csv")
papers_dataset = pd.read_csv("papers.csv")

authors_dataset.dropna(axis=0, how='any', inplace=True)
paper_authors_dataset.dropna(axis=0, how='any', inplace=True)
papers_dataset.dropna(axis=0, how='any', inplace=True)

print("authors_dataset")
print(authors_dataset.head(1))
print("paper_authors_dataset")
print(paper_authors_dataset.head(1))
print("papers_dataset")
print(papers_dataset.head(1))
print("papers_dataset **")
print(papers_dataset["paper_text"].head(1))

train_papers_test = pd.read_csv("train_papers.csv")
print("train_papers")
print(train_papers_test.head(2))

train_papers, validate_papers, test_papers = train_validate_test_split(papers_dataset)
train_papers.to_csv('train_papers.csv', index = False)
validate_papers.to_csv('validate_papers.csv', index = False)
test_papers.to_csv('test_papers.csv', index = False)

train_authors, validate_authors, test_authors = train_validate_test_split(authors_dataset)
train_authors.to_csv('train_authors.csv', index = False)
validate_authors.to_csv('validate_authors.csv', index = False)
test_authors.to_csv('test_authors.csv', index = False)