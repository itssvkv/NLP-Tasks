# from nltk import ngrams
# import nltk
# import pandas as pd

# dataset_path = "dataset/Emotion_classify_Data.csv"


# def read_dataset(path):
#     return pd.read_csv(path, dtype=str)


# def get_ngrams(text, n):
#     tokens = nltk.word_tokenize(text)
#     n_grams = list(ngrams(tokens, n))
#     return n_grams


# def process():
#     n = int(input("Enter number of N: "))
#     data_frame = read_dataset(dataset_path)
#     for _, row in data_frame.iterrows():
#         n_grams = get_ngrams(row["Comment"], n)
#         print(f"Text: {row['Comment']}")
#         print(f"{n}-grams: {n_grams}\n")


# process()

import pandas as pd
from nltk import ngrams, word_tokenize


def load_data_from_csv(file_path):
    return pd.read_csv(file_path, dtype=str)


def extract_ngrams(text, n_value):

    tokens = word_tokenize(text)
    ngrams_list = list(ngrams(tokens, n_value))
    return ngrams_list


def process_comments():
    n_gram_size = int(input("Enter the desired N-gram size: "))
    data = load_data_from_csv("dataset/Emotion_classify_Data.csv")

    for _, row in data.iterrows():
        comment_text = row["Comment"]
        n_grams = extract_ngrams(comment_text, n_gram_size)

        print(f"Text: {comment_text}")
        print(f"{n_gram_size}-grams: {n_grams}\n")


process_comments()
