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
