from dataclasses import dataclass
import string
import xmltodict
import os
from tqdm import tqdm
from typing import List, Set
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

config = {
    "data_dir": "./nyt_corpus/samples_500/",
    "use_lemmatizer": True
}

@dataclass
class News:
    text: str
    stemmed_words: List[str]
    catagory: Set[str]
    time: List[int] # Year, Month, Day

    bag_of_words: List[int]

@dataclass
class DictionaryEntry:
    ind: int
    freq: int

corpus: List[News] = []
dictionary: dict[str: DictionaryEntry] = {} # This dictionary includes word frequency

def parse_xml(xml: str, file_name: str) -> News:
    parsed_dict = xmltodict.parse(xml)["nitf"]
    text, stemmed_words, catagory, time = "", [], set(), [0, 0, 0]
    expected_text_absent = False

    # Get the full text
    try:
        text_nodes_root = parsed_dict["body"]["body.content"]
        if text_nodes_root is None: # It may happen, will not handle it as exception
            expected_text_absent = True
            text = None
        else:
            text_nodes = text_nodes_root["block"]
            full_text_node = None
            try:
                _ = text_nodes[1] # It will fail when there is only one 'block' node
            except:
                text_nodes = [text_nodes]

            for node in text_nodes:
                if node["@class"] == "full_text":
                    full_text_node = node
                    break
            assert full_text_node is not None
            text = " ".join(full_text_node["p"])
    except Exception as e:
        text = None
        print("Parse full text of %s failed." % file_name)
        print("Exception: %s" % str(e), end = '\n\n')

    # Process full text
    if text is None:
        stemmed_words = None
        if not expected_text_absent:
            print("Due to previous exception stemmation skipped.", end = '\n\n')
    else:
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words("english")
        text = text.lower().translate(str.maketrans("", "", string.digits + string.punctuation))
        stemmed_words = [
            (lemmatizer.lemmatize(word) if config["use_lemmatizer"] else stemmer.stem(word))
                for word in nltk.word_tokenize(text) if word not in stop_words
        ]

    # Get catagory
    try:
        catagory_nodes = \
            parsed_dict["head"]["docdata"]["identified-content"]["classifier"]
        try:
            _ = catagory_nodes[1] # It will fail when there is only one 'classifier' node
        except:
            catagory_nodes = [catagory_nodes]
        
        for node in catagory_nodes:
            res = node["#text"].split("/")
            if len(res) < 3:
                continue
            if res[0] == "Top" and (res[1] == "News" or res[1] == "Features"):
                catagory.add(res[2])
    except Exception as e:
        catagory = None
        print("Parse catagory of %s failed." % file_name)
        print("Exception: %s" % str(e), end = '\n\n')

    # Get date
    try:
        date_nodes = parsed_dict["head"]["meta"]
        for node in date_nodes:
            if node["@name"] == "publication_year":
                time[0] = int(node["@content"])
            elif node["@name"] == "publication_month":
                time[1] = int(node["@content"])
            elif node["@name"] == "publication_day_of_month":
                time[2] = int(node["@content"])
    except Exception as e:
        time = None
        print("Parse time of %s failed." % file_name)
        print("Exception: %s" % str(e), end = '\n\n')

    return News(
        text = text,
        stemmed_words = stemmed_words,
        catagory = catagory,
        time = time,
        bag_of_words = [] # This will be filled in later
    )

def construct_dict() -> None:
    global dictionary
    for news in corpus:
        if news.stemmed_words is None:
            continue
        for word in news.stemmed_words:
            if word not in dictionary:
                dictionary[word] = DictionaryEntry(
                    ind = len(dictionary),
                    freq = 1
                )
            else:
                dictionary[word].freq += 1

def construct_bag_of_words() -> None:
    for news in corpus:
        if news.stemmed_words is None:
            news.bag_of_words = None
            continue
        bag_of_words = [0 for _ in range(len(dictionary))]
        for word in news.stemmed_words:
            bag_of_words[dictionary[word].ind] += 1
        news.bag_of_words = bag_of_words

def draw_word_cloud() -> None:
    word_list = [(word, dictionary[word].freq) for word in dictionary]
    word_list.sort(key = lambda x: -x[1])
    word_list = { x[0]: x[1] for x in word_list[: 100] }

    plt.figure()
    plt.axis("off")
    plt.imshow(WordCloud(background_color = "white").generate_from_frequencies(word_list))
    plt.savefig("word-cloud.png")

def draw_word_len_hist() -> None:
    plt.figure()
    plt.hist([len(word) for word in dictionary])
    plt.savefig("word-len-hist.png")

def draw_word_cnt_hists() -> None:
    word_cnt = [len(news.stemmed_words)
        for news in corpus if news.stemmed_words is not None]

    # Equal-width
    depth, boundary = pd.cut(word_cnt, 10, retbins = True)
    plt.figure()
    plt.bar(
        (boundary[1: ] + boundary[: -1]) / 2, depth.value_counts(),
        width = boundary[1: ] - boundary[: -1] - 10)
    plt.savefig("word-cnt-width.png")

    # Equal-depth
    depth, boundary = pd.qcut(word_cnt, q = 10, retbins = True)
    plt.figure()
    plt.bar(
        (boundary[1: ] + boundary[: -1]) / 2, depth.value_counts(),
        width = boundary[1: ] - boundary[: -1] - 10)
    plt.savefig("word-cnt-depth.png")
    
def draw_news_catagory() -> None:
    catagory_dict = {}
    for news in corpus:
        if news.catagory is None:
            continue
        for cat in news.catagory:
            if cat not in catagory_dict:
                catagory_dict[cat] = 1
            else:
                catagory_dict[cat] += 1
    catagory_list = [(cat, catagory_dict[cat]) for cat in catagory_dict]
    catagory_list.sort(key = lambda x: -x[1])

    plt.figure()
    plot = plt.subplot(1, 1, 1)
    pos = plot.get_position()
    plot.set_position([
        pos.x0, pos.y0 + pos.height * 0.3,
        pos.width, pos.height * 0.75
    ])
    plt.bar(
        [x[0] for x in catagory_list],
        [x[1] for x in catagory_list]
    )
    plt.xticks([x[0] for x in catagory_list], rotation = 90)
    plt.savefig("news-catagory-bar.png")

def draw_news_month() -> None:
    month_cnt = [0 for _ in range(12)]
    for news in corpus:
        if news.time is None:
            continue
        month_cnt[news.time[1] - 1] += 1

    plt.figure()
    plt.bar(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        month_cnt
    )
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    plt.savefig("news-month-bar.png")

def main() -> None:
    global corpus
    for file_name in tqdm(os.listdir(config["data_dir"])):
        handler = open(
            os.path.join(config["data_dir"], file_name),
            encoding = "utf-8", mode = "r"
        )
        content = "\n".join(handler.readlines())
        corpus.append(parse_xml(content, file_name))
        handler.close()

    construct_dict()
    construct_bag_of_words()

    draw_word_cloud()
    draw_word_len_hist()
    draw_word_cnt_hists()

    draw_news_catagory()
    draw_news_month()

if __name__ == "__main__":
    main()