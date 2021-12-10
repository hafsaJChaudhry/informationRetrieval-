import re
from bs4 import BeautifulSoup
from os import walk
from tqdm import tqdm
import sys
import time
import threading

#i used this way
file = open('allFiles.txt', 'r')
book = file.read()

#you can also use this way to open all the files but i couldnt get it to work properly. was not finding my path
'''path = "/PycharmProjects/476phase1/files"
#gets each file in the directory of html files
files = os.listdir(path)

for file in sorted(files):
    directory = path + file
    url = open(directory, 'r')
    urlFinal = url.read()
    tmpText = h.handle(urlFinal)
    for tokens in tmpText:
        token = ''.join(char for char in tokens if char.isalpha())
        if token:
            # print(token)
            allTokens.append(token)
            output.write(str(token))
            output.write(" ")
    url.close()
    output.close()
    '''


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

def tokenize():
    if book is not None:
        words = book.lower().split()
        return words
    else:
        return None


def map_book(tokens):
    hash_map = {}

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

#get rid of special characters in text and lower case. only keep A-Z and a-z chars
book = re.sub('[^A-Za-z\s]+', '', book)
book = book.lower()

words = tokenize()
char_count = []

#put text in a list
word_list = " ".join(re.findall("[a-zA-Z]+", book))
word_list = re.findall('[\w]+', book)


# Create a Hash Map (Dictionary)
map = map_book(words)
count = 1
#sort numerically from greatest to least
highNum = sorted(map.items(), key = lambda x: x[1], reverse=True)
print("first 50 from greatest to least: ")
for word in highNum[:50]:
    print(count, word)
    count += 1
#print(highNum)
print()
count = 1
#sort numerically from least to greatest
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
#print(alpha2)

# Show Word Information
#for word in word_list:
   #print('Word: [' + word + '] Frequency: ' + str(map[word]))