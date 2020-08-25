from __future__ import print_function
from torch import nn
from torchtext import data
import spacy
import re
import nltk

class TextPreprocessing(nn.Module):
    def __init__(self):
        super().__init__()
        self.spacy_en = spacy.load('en_core_web_sm')
        self.TEXT = data.Field(sequential=True, tokenize=self.tokenizer, lower=True, pad_first=True, fix_length=10)
        self.LABEL = data.Field(sequential=False, use_vocab=False)

    def tokenizer(self, text):
        text = text.replace("\n", " ")
        return [token.text for token in self.spacy_en.tokenizer(text) if not token.is_punct and not token.is_digit and not token.like_num]

    def build_text(self):
        # train, val, test = data.TabularDataset.splits(
        #         path='', train='train_papers.csv', skip_header=True,
        #         validation='validate_papers.csv', test='test_papers.csv', format='csv',
        #         fields=[('id', self.LABEL), ("year", self.LABEL), ("titel", self.TEXT), ("event_type", self.TEXT), ("pdf_name", self.TEXT), ("abstract", self.TEXT), ("paper_text", self.TEXT)])


        # train, val, test = data.TabularDataset.splits(
        #         path='', train='train_authors.csv', skip_header=True,
        #         validation='validate_authors.csv', test='test_authors.csv', format='csv',
        #         fields=[('id', self.TEXT), ("name", self.TEXT)])
        #
        # print(vars(train[0]))

        train, val, test = data.TabularDataset.splits(
                path='', train='dummy_train.csv', skip_header=True,
                validation='dummy_val.csv', test='dummy_test.csv', format='csv',
                fields=[('sents', self.TEXT)])
        return train, val, test

    def build_vocabulary(self):
        train, _, _ = self.build_text()
        self.TEXT.build_vocab(train, vectors="glove.6B.100d")
        vocab = self.TEXT.vocab
        emb_dim = len(vocab.vectors[0])
        embed = nn.Embedding(len(vocab), emb_dim)
        embed.weight.data.copy_(vocab.vectors)
        # print("inds")
        # print(vocab.stoi["oranges"])
        # print(vocab.stoi["books"])
        # print(vocab.stoi["apples"])
        return vocab, embed

    def build_iterator(self, train, val, test):
        train_iter, val_iter, test_iter = data.Iterator.splits(
            (train, val, test), sort_key=lambda x: len(x.Text),
            batch_sizes=(10, 5, 5))  # Torchtext already uses dynamic padding
        return train_iter, val_iter, test_iter

preprocessing = TextPreprocessing()
train, val, test = preprocessing.build_text()
vocab, embed =  preprocessing.build_vocabulary()
# orig_text = vocab.itos[12]
# print(orig_text)
print("len = " + str(len(vocab.itos)))
#
# for i in range(19):
#     print(vars(train[i]))
# print("prining train")
# print(train)

train_iter, val_iter, test_iter = preprocessing.build_iterator(train, val, test)

print("Train iter", next(iter(train_iter)))

for val_i, data in enumerate(train_iter):
  sent = data.sents
  print(data)
  print(val_i)
  print(sent)
  # print(x)
  # print(x_lengths)