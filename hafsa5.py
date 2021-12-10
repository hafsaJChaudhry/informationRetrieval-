'''
phase 5: continuation of previous phases
'''
from os import walk
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import sys, os, time, html, math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
INPUT_DIR_PATH = './' + INPUT_DIR
OUT_DIR_PATH = './' + OUTPUT_DIR
#DOC_NUMBERS_TO_PLOT = [10, 20, 40, 80, 100, 200, 300, 400, 500]

tfidf_vectorizer = TfidfVectorizer()
f = open("stopwords.txt", "r")
stopwords = np.loadtxt(f, dtype=str)
f.close()

def main():
    file_list = os.listdir(INPUT_DIR)
    documents = {}

    documents = extract_data(file_list)
    print("Calculating similarity....takes roughly 30 mins:")
    similarity_matrix = file_similarity(documents)
    c_clusters = get_clusters(similarity_matrix, set(), -1)
    closest_documents = c_clusters[0]
    c_output = closest_documents[1] + ' and ' + closest_documents[2]
    print("The Closest Documents: ", c_output)
    print("Similarity is: ", closest_documents[0])

    f_clusters = get_clusters(similarity_matrix, set(), 5000)
    farthest_documents = f_clusters[1]
    f_output = farthest_documents[1] + ' and ' + farthest_documents[2]
    print("The Farthest Documents: ", f_output)
    print("Similarity is: ", farthest_documents[0])

    #centroid
    centroids = {}
    for fl in documents:
        centroids[fl] = [fl]
    print()
    kmeans_cluster(centroids, similarity_matrix)

def file_similarity(document_data):
    cos_sim_matrix = {}

    for file_1 in document_data:
        cos_sim_matrix[file_1] = {}
        for file_2 in document_data:

            if file_1 == file_2:
                break

            if not len(document_data[file_1].strip()) and not len(document_data[file_2].strip()):
                break

            tfidf_matrix = tfidf_vectorizer.fit_transform([document_data[file_1],document_data[file_2]])
            cos_sim_matrix[file_1][file_2] = ((tfidf_matrix * tfidf_matrix.T).A)[0, 1]

    return cos_sim_matrix

# Get list of all input files from the directory
def get_file_list():
    file_list = []
    if not os.path.exists(INPUT_DIR_PATH):
        print("Input directory does not exist")
    else:
        for (dirpath, dirnames, filenames) in walk(INPUT_DIR_PATH):
            file_list.extend(filenames)
            return file_list

# Create the output Directory if it doesn't exist
def validate_output_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        print("Directory with name ", OUTPUT_DIR, " created.")
    else:
        print("Directory with name ", OUTPUT_DIR, " already exists.")

def get_clusters(data, inactive, num):
    max_num = -1
    min_num = 5000
    maximum = (max_num, 0, 0)
    minimum = (min_num, 0, 0)

    for d1 in data.keys():
        for d2 in data[d1].keys():

            if d1 not in inactive and d2 not in inactive and d1 != d2:
                score = data[d1][d2]

                if num == -1:
                    if score > max_num:
                        max_num = score
                        maximum = (max_num, d1, d2)

                elif( num == 5000):
                    if score < min_num:
                        min_num = score
                        minimum = (min_num, d1, d2)

    return maximum, minimum

def calculate_doc_freq_per_word(global_word_counter, word_freq_per_doc):
    doc_freq_per_word = {}
    for word in global_word_counter:
        count = 0
        for elem in word_freq_per_doc.items():
            words = elem[1]
            if word in words.keys():
                count += 1

        if count > 0:
            doc_freq_per_word[word] = count

    return doc_freq_per_word

def display_line_graph(xaxis, proc_times_cumulative):
    plt.title("Processing times vs number of documents processed")
    plt.ylabel('Time in milliseconds')
    plt.xlabel('Number of files')
    plt.grid(True)
    plt.plot(xaxis, proc_times_cumulative)

    plt.savefig(OUT_DIR_PATH + '/time_vs_no_files.png')
    plt.show()

def extract_data(file_list):
    document_data = {}
    for fl in tqdm(file_list):

        path = INPUT_DIR + '/' + fl
        f = open(path, 'r', encoding="utf-8", errors='ignore')

        try:
            file_data = f.read()
            soup = BeautifulSoup(file_data, 'html.parser')
            html_data = soup.get_text()
            html_data = html.unescape(html_data).lower()
            extracted_data = " ".join(re.findall("[a-zA-Z]+", html_data))

            for stopword in stopwords:
                extracted_data = extracted_data.replace(" " + stopword + " ", " ")

            document_data[fl] = extracted_data

        finally:
            f.close()
    return document_data

def kmeans_cluster(clusters, data):
    inactive = set()
    total_clusters = len(clusters.keys())

    while total_clusters - len(inactive) - 1:
        cname = str(total_clusters)
        c_clusters = get_clusters(data, inactive, -1)
        cosine_score, c1, c2 = c_clusters[0]

        if cosine_score != -1:
            new_cluster = [c1, c2]
            inactive.update(new_cluster)
            clusters[cname] = new_cluster
            data[cname] = {}

            for cluster in clusters:

                if cluster not in inactive:
                    cluster1 = clusters[cname]
                    cluster2 = clusters[cluster]
                    document_data = len(cluster1) + len(cluster2)
                    score = 0

                    for doc1 in cluster1:
                        for doc2 in cluster2:

                            if doc1 in data and doc2 in data[doc1]:
                                score += data[doc1][doc2]
                            if doc2 in data and doc1 in data[doc2]:
                                score += data[doc2][doc1]

                        data[cname][cluster] = score / document_data

            total_clusters += 1
        print("cluster pair " + c1 + "+" + c2 + "==" + cname + "=" + str(cosine_score))



main()