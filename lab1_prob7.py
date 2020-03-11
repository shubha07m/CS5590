import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
import random

random.seed(123)

# Reading the raw text file #
mydata = open('nlp_input.txt')
f = mydata.read()

# Tokenizing and applying Lemmatization technique #
stokens = nltk.sent_tokenize(f)
wtokens = nltk.word_tokenize(f)
Lemmatizer = WordNetLemmatizer()
for word in wtokens:
    print("Lemmatized word is : "+Lemmatizer.lemmatize(word))

# Finding all the trigrams for the words #
trigram_word = ngrams(wtokens, 3)

# Showing the top 10 most repeated trigrams#

word_freq_dist = nltk.FreqDist(trigram_word)
ten_most_tri = (word_freq_dist.most_common(10))
print("Top 10 of the most repeated trigrams based on their count are:")
print(ten_most_tri)
tri_list = []
for triagram in ten_most_tri:
    tri_list.append(''.join(triagram[0]))
x = (tri_list[0])
a = []
for sent in stokens:
    for tri in tri_list:
        if tri in ("".join(sent.split())):
            a.append(sent)
print("The final concatenated result is:")
a = ''.join(list(set(a)))
print(a)
