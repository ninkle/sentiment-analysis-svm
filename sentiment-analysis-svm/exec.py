from sklearn.externals import joblib
import string
from nltk.stem import SnowballStemmer
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.stem import SnowballStemmer
from nltk.chunk import tree2conlltags
from sklearn.feature_extraction.text import TfidfVectorizer


def clean(s):
    s = s.strip(string.punctuation)
    # initialize snowball stemmer
    stemmer = SnowballStemmer("english")
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
    return cleaned



def process(text):
    data = []
    text = clean(text)
    data.append(text)
    return data


if __name__ == '__main__':
    clf = joblib.load('classifier.pkl')
    vec = joblib.load('vectorizer.pkl')
    cond = True
    while cond == True:
        x = input("Enter some example text:")
        if x == "q":
            cond = False
        else:
            x = process(x)
            print(x)
            x_test = vec.transform(x)
            print(clf.predict(x_test))