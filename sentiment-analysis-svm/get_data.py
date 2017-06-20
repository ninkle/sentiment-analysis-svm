import os
import pandas as pd
import glob
import re
import string

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.stem import SnowballStemmer
from nltk.chunk import tree2conlltags
from nltk.util import ngrams
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

contractions_dict = {
    'd': 'had',
    'don': 'do',
    'didn': 'did',
    'll': 'will',
    'm': 'am',
    've': 'have',
    're': 'are',
    's': 'is',
    't': 'not',
    'wouldn': 'would'
}

def clean(s):
    s = s.strip(string.punctuation)
    # initialize snowball stemmer
    stemmer = SnowballStemmer("english")
    # initialize vectorizer
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 3))
    # pos tags
    tree = (ne_chunk(pos_tag(word_tokenize(s))))
    tags = tree2conlltags(tree)
    words = []
    for i in tags:
        if i[1] != ("NNP" or "PRP"):
            # get word stem
            words.append(stemmer.stem(i[0]))
        else: # special case
            # replace proper nouns and personal pronouns with pos tag
            words.append(stemmer.stem(i[1]))
    cleaned = " ".join(words)
    analyze = ngram_vectorizer.build_analyzer()
    ngrams = analyze(cleaned)
    return ngrams


def process(label):
    data = []
    print("Begin process for %s label." % label)
    count = 0
    for filename in glob.iglob(os.path.join('txt_sentoken', label, '*.txt')):
        if count % 100 == 0:
            print("On file %i of %s label" %(count, label))
        f = open(filename, "r")
        text = f.read()
        text = clean(text)
        for i in text:
            data.append(i)
        count += 1
    return data

def process_tweets(label):
    print("And now for the tweets")
    tweets = []
    twitter_words = ['rt', 'RT', 'dm', 'DM', 'MKR', 'mkr']
    for filename in glob.iglob(os.path.join(label, '*.txt')):
        f = open(filename, "r")
        text = f.read()
        text = text.split()
        text = [i for i in text if i not in twitter_words]
        text = ' '.join(text)
        pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
        result = pattern.sub(lambda x: contractions_dict[x.group()], text)
        result = clean(result)
        for i in result:
            tweets.append(i)
    return tweets

neg = process('neg')
pos = process('pos')
neutral = process_tweets('None')

print(neutral[:10])

neg_data = joblib.dump(neg, 'neg_data.pkl')
pos_data = joblib.dump(pos, 'pos_data.pkl')
neutral_data = joblib.dump(neutral, 'neutral_data.pkl')