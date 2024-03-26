from nltk import ngrams
import nltk
import pandas as pd

dataset_path = "dataset/Emotion_classify_Data.csv"


def read_dataset(path):
    return pd.read_csv(path, dtype=str)


def get_ngrams(text, n):
    tokens = nltk.word_tokenize(text)
    n_grams = list(ngrams(tokens, n))
    return n_grams


def process():
    n = int(input("Enter number of N: "))
    data_frame = read_dataset(dataset_path)
    for _, row in data_frame.iterrows():
        n_grams = get_ngrams(row["Comment"], n)
        print(f"Text: {row['Comment']}")
        print(f"{n}-grams: {n_grams}\n")


process()
