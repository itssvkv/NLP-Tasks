import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer as p_stem
from nltk.stem.snowball import SnowballStemmer as s_stem

dataset_path = "dataset/Emotion_classify_Data.csv"
porter_results_path = "output/porter_output.csv"
snowball_results_path = "output/snowball_output.csv"


def read_dataset(path):
    return pd.read_csv(path)


def tokenize_quote(quote):
    return word_tokenize(quote)


def stem_by_porter(words):
    result = [p_stem().stem(word) for word in words]
    return result


def stem_by_snowball(words):
    result = [s_stem(language="english").stem(word) for word in words]
    return result


def process():
    data_frame = read_dataset(dataset_path)
    data_frame["porter_stem_output"] = (
        data_frame["Comment"].apply(tokenize_quote).apply(stem_by_porter)
    )
    data_frame["snowball_stem_output"] = (
        data_frame["Comment"].apply(tokenize_quote).apply(stem_by_snowball)
    )
    print(data_frame["porter_stem_output"])

    data_frame["porter_stem_output"].to_csv(porter_results_path, index=False)
    data_frame["snowball_stem_output"].to_csv(snowball_results_path, index=False)


process()
