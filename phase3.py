'''continuation of hw2 (same terms and term weights)

goal: build a pair of files of *fixed length records*
output1: dictionary w/ token, # of docs that have token,
and location of first instance

output2: contains document id, and normalized weight of the word
in the document
'''
from os import walk
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import sys, os, time, html, math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

OUTPUT_DIR = sys.argv[2]
INPUT_DIR_PATH = './' + sys.argv[1]
OUT_DIR_PATH = './' + OUTPUT_DIR
DOC_NUMBERS_TO_PLOT = [10, 20, 40, 80, 100, 200, 300, 400, 500]

def main():
    # Dictionary to maintain word frequency across documents
    word_freq_per_doc = {}
    doc_len = {}

    #word_freq_per_doc = 340.html
    #doc_freq_per_word = nezmet

    # List to store time taken to process each file
    proc_times_cumulative = [0]

    file_list = get_file_list()  # file list with file names
    validate_output_directory() # creates output directory

    #type = 'collections.Counter'
    global_word_counter = extract_data(file_list, word_freq_per_doc, doc_len, proc_times_cumulative) #extracts text, times and plot

    # first key should by nemzet, first value should be 430, position should be 1
    create_dictionary_output(global_word_counter[0])
    create_postings_output(global_word_counter[1], global_word_counter[2])

    #graph stuff
    proc_times_cumulative.pop(0)
    display_line_graph(DOC_NUMBERS_TO_PLOT, proc_times_cumulative)

# output file 1
def create_dictionary_output(global_word_counter):
    #create dictionary.txt file
    dictionary_file_path = OUT_DIR_PATH + "/dictionary.txt"
    f = open(dictionary_file_path, "w")
    previous_value = 0
    previous_count = 0
    counter = 0
    for key, value in global_word_counter.items():
        if counter == 0:
            f.write(key + "\n" + str(value) + "\n")
            counter += 1
            f.write(str(counter) + "\n")
            previous_value = value
            previous_count = counter
        else:
            f.write("\n"+ key + "\n" + str(value) + "\n")
            counter = previous_value + previous_count
            f.write(str(counter) + "\n")
            previous_value = value
            previous_count = counter
    f.close()

# output file 2
def create_postings_output(doc_freq_per_word, term_freq_and_idf):
    postings = []

    for word in doc_freq_per_word:
        html_id = doc_freq_per_word[word]
        for id in html_id:
            weight = term_freq_and_idf[id][word]
            postings.append((id, weight))

    postings_file_path = OUT_DIR_PATH + "/postings.txt"

    f = open(postings_file_path, "w")
    for id, term_freq_and_idf in postings:
        f.write(id + "\t" + str(term_freq_and_idf) + "\n")
    f.close()

# Get list of all input files from the directory
def get_file_list():
    file_list = []
    if not os.path.exists(INPUT_DIR_PATH):
        print("Input directory does not exist")
    else:
        for (dirpath, dirnames, filenames) in walk(INPUT_DIR_PATH):
            file_list.extend(filenames)
            return file_list

def extract_data(file_list, word_freq_per_doc, doc_len, proc_times_cumulative):
    global postings, term_freq_and_idf
    s = set()
    global_word_counter = Counter()     #  a new empty counter
    doc_freq_per_word = {}
    term_freq = {}

    f = open("stopWords.txt", "r")
    stopwords = np.loadtxt(f, dtype=str)
    f.close()

    # start recording time
    start_time = time.time()
    file_count = 0

    for fl in tqdm(file_list):
        file_count += 1
        wordcount = Counter()

        path = INPUT_DIR_PATH + '/' + fl
        f = open(path, 'r', encoding="utf-8", errors='ignore')

        try:
            file_data = f.read()
            soup = BeautifulSoup(file_data, 'html.parser')
            html_data = soup.get_text()
            html_data = html.unescape(html_data).lower()
            extracted_data = " ".join(re.findall("[a-zA-Z]+", html_data))

            extracted_data = extracted_data.split(' ')

            for word in extracted_data:
                if word not in stopwords and len(word) > 1:
                    if word in global_word_counter:
                        global_word_counter.update({word: 1})
                    elif word in s:
                        global_word_counter.update({word: 2})
                    else:
                        s.add(word)

                    wordcount.update({word: 1})

            # stop recording time
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000

            if file_count in DOC_NUMBERS_TO_PLOT:
                proc_times_cumulative.append(proc_times_cumulative[-1] + processing_time)

        finally:
            f.close()

        word_freq_per_doc[fl] = wordcount
        # contains weight of word for each doc
        term_freq[fl] = {}

        for word in word_freq_per_doc[fl]:

            # document freq hash map
            if word not in doc_freq_per_word:
                doc_freq_per_word[word] = set()

            doc_freq_per_word[word].add(fl)

            # weight of each word
            denominator = sum(word_freq_per_doc[fl].values())
            numerator = word_freq_per_doc[fl][word]
            word_term_freq = numerator / denominator

            term_freq[fl][word] = word_term_freq

        doc_len[fl] = sum(wordcount.values())

        # calculate idf and weights of words
        term_freq_and_idf = {}

        for file in word_freq_per_doc:
            words = word_freq_per_doc[file]
            term_freq_and_idf[file] = {}
            for word in words:
                idf = math.log(file_count / len(doc_freq_per_word[word]))
                term_freq_and_idf[file][word] = term_freq[file][word] * idf

    return global_word_counter, doc_freq_per_word, term_freq_and_idf

# Create the output Directory if it doesn't exist
def validate_output_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        print("Directory with name ", OUTPUT_DIR, " created.")
    else:
        print("Directory with name ", OUTPUT_DIR, " already exists.")

def display_line_graph(xaxis, proc_times_cumulative):
    plt.title("Processing times vs number of documents processed")
    plt.ylabel('Time in milliseconds')
    plt.xlabel('Number of files')
    plt.grid(True)
    plt.plot(xaxis, proc_times_cumulative)

    plt.savefig(OUT_DIR_PATH + '/time_vs_no_files.png')
    plt.show()

main()