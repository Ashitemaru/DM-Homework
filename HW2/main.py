config = {
    "doc_num": 300,
    "doc_folder_name": "nyt_corp0",
    "debug_file": "test.txt",
    "tfidf_file": "tfidf.txt",
    "co_matrix_file": "comat.txt"
}

corpus = []

dictionary = {} # The dict mapping word to index
reverse_dictionary = [] # The list mapping index to word

tf_matrix = [] # TF
idf_map = [] # IDF
tfidf_matrix = [] # TF-IDF

co_matrix = [] # Co-occurence matrix

import math
from tqdm import tqdm
import os

def debug(obj: object) -> None:
    debug_handler = open(config["debug_file"], "w")
    debug_handler.write(str(obj))
    debug_handler.close()

def get_input_int(caption: str, range: int) -> int:
    res = -1
    while True:
        input_str = input(caption + " ")
        try:
            res = int(input_str)
        except:
            print("The input is not a number.")
            continue
        if 0 <= res < range:
            return res
        else:
            print("The input number is out of range.") 

def get_input_str(caption: str, range: list) -> str:
    while True:
        input_str = input(caption + " ")
        if input_str in range:
            return input_str.replace("\n", "")
        else:
            print("The input is out of range.")  

def eu_distance(x: list, y: list) -> float:
    if len(x) != len(y):
        return -1
    else:
        s = 0
        for ind, ele in enumerate(x):
            s += (ele - y[ind]) ** 2
        return math.sqrt(s)

def cos_distance(x: list, y: list) -> float:
    if len(x) != len(y):
        return -1
    else:
        x_len = 0
        for ele in x:
            x_len += ele ** 2
        x_len = math.sqrt(x_len)

        y_len = 0
        for ele in y:
            y_len += ele ** 2
        y_len = math.sqrt(y_len)

        product = 0
        for ind, ele in enumerate(x):
            product += ele * y[ind]
        return 1 - product / (x_len * y_len)

def read_corp() -> None:
    print("Now reading in corpus")
    global corpus
    for ind in tqdm(range(config["doc_num"])):
        file_path = "./%s/%d" % (config["doc_folder_name"], ind)
        file_handle = open(file_path, "r")
        doc = file_handle.readlines()
        doc = ' '.join(doc) # Concat all the lines
        doc = ''.join([
            char for char in doc
                if char.isalnum() or char == ' '
        ]) # Filter out punctuations
        word_list = doc.split(' ') # Split by space
        word_list = [
            word.lower() for word in word_list
                if len(word) != 0
        ] # Filter empty words & trans to lower
        corpus.append(word_list)

        file_handle.close()

def construct_dict() -> None:
    print("Now constructing dictionary")
    global dictionary, reverse_dictionary
    for word_list in corpus:
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
                reverse_dictionary.append(word)

def calc_tf() -> None:
    print("Now calculating TF")
    global tf_matrix
    tf_matrix = [
        [0 for _ in range(len(dictionary))]
            for __ in range(len(corpus))
    ]

    for ind, doc in tqdm(enumerate(corpus)):
        for word in doc:
            tf_matrix[ind][dictionary[word]] += 1
        tf_matrix[ind] = [n / len(doc) for n in tf_matrix[ind]]

def calc_idf() -> None:
    print("Calculating IDF")
    global idf_map
    idf_map = [0 for _ in range(len(dictionary))]
    for word_ind, word in tqdm(enumerate(reverse_dictionary)):
        for doc in corpus:
            if word in doc:
                idf_map[word_ind] += 1
    idf_map = [math.log10(len(corpus) / (1 + x)) for x in idf_map]

def calc_tfidf() -> None:
    print("Now merge TF & IDF")
    global tfidf_matrix
    tfidf_matrix = [
        [0 for _ in range(len(dictionary))]
            for __ in range(len(corpus))
    ]
    for ind in tqdm(range(len(corpus))):
        tfidf_matrix[ind] = [
            tf * idf_map[word_ind]
                for word_ind, tf in enumerate(tf_matrix[ind])
        ]

def calc_co_matrix() -> None:
    print("Calculating co-occurence matrix")
    global co_matrix
    co_matrix = [
        [[] for _ in range(len(dictionary))]
            for __ in tqdm(range(len(dictionary)))
    ]
    print("Constructing initial matrix finished")
    for doc_ind, doc in tqdm(enumerate(corpus)):
        for i in range(len(doc)):
            for j in range(len(doc)):
                i_ind = dictionary[doc[i]]
                j_ind = dictionary[doc[j]]
                if len(co_matrix[i_ind][j_ind]) == 0 or co_matrix[i_ind][j_ind][-1] != doc_ind:
                    co_matrix[i_ind][j_ind].append(doc_ind)
    for i in range(len(dictionary)):
        for j in range(len(dictionary)):
            co_matrix[i][j] = len(co_matrix[i][j])

def doc_similarity_query(ind: int) -> None:
    # Euclidean distance
    eu_list = [
        (other_ind, eu_distance(tfidf_matrix[ind], tfidf_matrix[other_ind]))
            for other_ind in range(len(corpus))
    ]
    eu_list.sort(key = lambda x: x[1])
    print("According to euclidean distance, results are:")
    print(eu_list[1: 6]) # A doc is always similar to itself

    # Cosine distance
    cos_list = [
        (other_ind, cos_distance(tfidf_matrix[ind], tfidf_matrix[other_ind]))
            for other_ind in range(len(corpus))
    ]
    cos_list.sort(key = lambda x: x[1])
    print("According to cosine distance, results are:")
    print(cos_list[1: 6]) # A doc is always similar to itself

def word_similarity_query(ind: int) -> None:
    # Euclidean distance
    eu_list = [
        (other_ind, eu_distance(co_matrix[ind], co_matrix[other_ind]))
            for other_ind in tqdm(range(len(dictionary)))
    ]
    eu_list.sort(key = lambda x: x[1])
    print("According to euclidean distance, results are:")
    print(list(map(lambda x: (reverse_dictionary[x[0]], x[1]), eu_list[1: 6])))

    # Cosine distance
    cos_list = [
        (other_ind, cos_distance(co_matrix[ind], co_matrix[other_ind]))
            for other_ind in tqdm(range(len(dictionary)))
    ]
    cos_list.sort(key = lambda x: x[1])
    print("According to cosine distance, results are:")
    print(list(map(lambda x: (reverse_dictionary[x[0]], x[1]), cos_list[1: 6])))

def main():
    read_corp()
    construct_dict()
    calc_tf()
    calc_idf()
    calc_tfidf()
    calc_co_matrix()
    
    query_doc_ind = get_input_int("Which doc will you query?", len(corpus))
    doc_similarity_query(query_doc_ind)

    query_word = get_input_str("Which word will you query?", reverse_dictionary)
    word_similarity_query(dictionary[query_word])

if __name__ == "__main__":
    main()