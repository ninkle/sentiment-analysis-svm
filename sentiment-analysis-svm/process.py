from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.stem import SnowballStemmer
from nltk.chunk import tree2conlltags
import time


def clean(s):
    # initialize snowball stemmer
    stemmer = SnowballStemmer("english")
    # pos tags
    tree = (ne_chunk(pos_tag(word_tokenize(s))))
    tags = tree2conlltags(tree)
    print(tags)
    result = []
    for i in tags:
        if i[1] != ("NNP" or "PRP"):
            # get word stem
            result.append(stemmer.stem(i[0]))
        else: # special case
            # replace proper nouns and personal pronouns with pos tag
            result.append(stemmer.stem(i[1]))
    result = " ".join(result)
    return result
