import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer

dataset_path = "task2/dataset/dataset.csv"
porter_results_path = "task2/output/porter_output.csv"
snowball_results_path = "task2/output/snowball_output.csv"


def read_dataset(path):
    return pd.read_csv(path)


def tokenize_quote(quote):
    return word_tokenize(quote)


def stem_by_porter(words):
    porter_stemmer = PorterStemmer()
    return [porter_stemmer.stem(word) for word in words]


def stem_by_snowball(words):
    snowball_stemmer = SnowballStemmer(language="english")
    return [snowball_stemmer.stem(word) for word in words]


def process():
    data_frame = read_dataset(dataset_path)
    data_frame["porter_stem_output"] = (
        data_frame["original_text"].apply(tokenize_quote).apply(stem_by_porter)
    )
    data_frame["snowball_stem_output"] = (
        data_frame["original_text"].apply(tokenize_quote).apply(stem_by_snowball)
    )
    print(data_frame["porter_stem_output"])

    data_frame["porter_stem_output"].to_csv(porter_results_path, index=False)
    data_frame["snowball_stem_output"].to_csv(snowball_results_path, index=False)


process()
