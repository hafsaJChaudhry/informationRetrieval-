'''
Author : Hafsa Chaudhry
Class: Information Retrieval
'''
import re
import os.path
import os,glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd

import nltk
from collections import defaultdict
import glob
from bs4 import BeautifulSoup
from os import walk
from tqdm import tqdm
from collections import Counter
import sys
import time
import threading


def openFile(fname):
    #for fname in os.listdir("files"):
    with open(os.path.join("files ", fname), encoding="iso8859-1") as f:
        text = f.read()
    return text


books = {}
for fname in os.listdir("files "):
    this_file_contents = openFile(fname)
    books[fname]=this_file_contents #adding a new entry to a dictionary.key is the filename, and the value is the string of its contents
book = '\n'.join(books.values()) #dictionary has a .values() function which returns a list of all the values without their keys.  so, a list of all the file contents
file2 = open('stopWords.txt', 'r')
book2 = file2.read()


class ElapsedTimeThread(threading.Thread):
    """"Stoppable thread that prints the time elapsed"""
    def __init__(self):
        super(ElapsedTimeThread, self).__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        thread_start = time.time()
        while not self.stopped():
            print("\rElapsed Time {:.3f} seconds".format(time.time()-thread_start), end="")
            #include a delay here so the thread doesn't uselessly thrash the CPU
            time.sleep(0.01)

if __name__ == "__main__":
    start = time.time()
    thread = ElapsedTimeThread()
    thread.start()
    # do something
    time.sleep(5)
    # something is finished so stop the thread
    thread.stop()
    thread.join()
    print() # empty print() to output a newline
    print("Finished in {:.3f} seconds".format(time.time()-start))

def tokenize(books):
    if books is not None:
        words = books.lower().split()
        return words
    else:
        return None

def map_book(tokens):
    hash_map = defaultdict(list)  #{}

    if tokens is not None:
        for element in tokens:
            # Remove Punctuation
            word = element.replace(",", "")
            word = word.replace(".", "")

            # Word Exist?
            if word in hash_map:
                hash_map[word] = hash_map[word] + 1
            else:
                hash_map[word] = 1

        return hash_map
    else:
        return None

def onlyOnce(list, c):
    count = Counter(list)
    return [word for word in list if count[word] >= c]


# get rid of special characters in text and html words, only keep A-Z and a-z chars. and lower case
book = re.sub('[^A-Za-z\s]+', '', book)
book = re.sub(r'\w*html\w*', '', book)
book = re.sub(r'<.*?>','', book)
book = book.lower()   #lower cases all A-Z chars

words = tokenize(book)
stopwords = tokenize(book2)
char_count = []


# update list w/ no stop words, no occurences less than once, and no words of length of 1
noStopWords = [x for x in words if x not in stopwords]
noOccurencesLessThanOnce = (onlyOnce(noStopWords, 2))
noSingleLetters = [word for word in noOccurencesLessThanOnce if len(word) >= 2]

# reassign words to updated list
words = noSingleLetters

# Create a Hash Map (Dictionary) with word frequency
map = map_book(words)
count = 1

'''  ----------------------------------PHASE ONE ---------------------------------------------
# sort numerically from greatest to least
highNum = sorted(map.items(), key = lambda x: x[1], reverse=True)
print("first 50 from greatest to least: ")
for word in highNum[:50]:
    print(count, word)
    count += 1
#print(highNum)

print()
count = 1
# sort numerically from least to greatest
lowNum = sorted(map.items(), key = lambda x: x[1])
print("first 50 from least to greatest: ")
for word in lowNum[:50]:
    print(count, word)
    count += 1
#print(lowNum)

print()
count = 1
print("first 50 alphabetically z-a: ")
alpha = sorted(map.items(), key=lambda x: x[0], reverse=True)
for word in alpha[:50]:
    print(count, word)
    count += 1
#print(alpha)

print()
count = 1
print("first 50 alphabetically a-z: ")
alpha2 = sorted(map.items(), key=lambda x: x[0])
for word in alpha2[:50]:
    print(count, word)
    count += 1
----------------------------------PHASE ONE --------------------------------------------- '''



'''----------------------------------PHASE TWO ---------------------------------------------'''

#modified from online tutorial
print("Term weight: ")
cvec = CountVectorizer(stop_words='english', min_df=3, max_df=0.5, ngram_range=(1,2))
sf = cvec.fit_transform(words)
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(10)
print(weights_df)
print("----------end of term weight----------")


print("Term weight for 002.html: ")
raw = open('/Users/Hafsa/PycharmProjects/476phase1/files /002.html',encoding="iso8859-1").read().lower()
raw = re.sub('[^A-Za-z\s]+', '', raw)
raw = re.sub(r'\w*html\w*', '', raw)
raw = raw.lower()
words = tokenize(raw)
raw1 = [x for x in words if x not in stopwords]
raw2 = (onlyOnce(raw1, 2))
raw3 = [word for word in raw2 if len(word) >= 2]
words = raw3
cvec = CountVectorizer(stop_words='english', min_df=3, max_df=0.5, ngram_range=(1,2))
sf = cvec.fit_transform(words)
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(10)
print(weights_df)


#modified from online tutorial
#inputFiles is the modified version of files that holds n amount of files per run (to time the input/output running time)
print("term weight w/ input per output: ")
path = '/Users/Hafsa/PycharmProjects/476phase1/inputFiles/*.html'
files = glob.glob(path)
chars = [file[3:] for file in files]
charList = []
charDict = {}
for char in chars:
    raw = open('/Us'+char,encoding="iso8859-1").read().lower()
    raw = re.sub('[^A-Za-z\s]+', '', raw)
    raw = re.sub(r'\w*html\w*', '', raw)
    raw = raw.lower()
    words = tokenize(raw)
    raw1 = [x for x in words if x not in stopwords]
    raw2 = (onlyOnce(raw1, 2))
    raw3 = [word for word in raw2 if len(word) >= 2]
    tokens = raw3
    text = nltk.Text(tokens, name=char)
    charList.append(text)
    charDict[char] = text
corpus = nltk.TextCollection(charList)
allWords = []
for char in charList:
    for token in char.tokens:
        if token not in allWords:
            allWords.append(token)
tfidfs = {}
for word in allWords:
    scores = []
    for char in chars:
        score = corpus.tf_idf(word, charDict[char])
        scores.append(score)
    tfidfs[word] = scores
df = pd.DataFrame(tfidfs, index=chars)
print(df)
#used to debug:
#print(word_list(highNum))


