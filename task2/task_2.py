import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer


def load_data(filepath):
    return pd.read_csv(filepath)


def split_into_words(text):
    return word_tokenize(text)


def apply_stemming(words, stemmer_type="porter"):
    if stemmer_type == "porter":
        stemmer = PorterStemmer()
    elif stemmer_type == "snowball":
        stemmer = SnowballStemmer(language="english")
    else:
        raise ValueError("Invalid stemmer type. Choose 'porter' or 'snowball'.")

    return [stemmer.stem(word) for word in words]


def process_data(data):
    data["porter_stemmed_comments"] = (
        data["Comment"]
        .apply(split_into_words)
        .apply(apply_stemming, stemmer_type="porter")
    )
    data["snowball_stemmed_comments"] = (
        data["Comment"]
        .apply(split_into_words)
        .apply(apply_stemming, stemmer_type="snowball")
    )
    return data


def save_results(data, porter_output_path, snowball_output_path):
    data["porter_stemmed_comments"].to_csv(porter_output_path, index=False)
    data["snowball_stemmed_comments"].to_csv(snowball_output_path, index=False)


data = load_data("dataset/Emotion_classify_Data.csv")
processed_data = process_data(data.copy())  # Avoid modifying the original data frame

print(processed_data["porter_stemmed_comments"])
save_results(
    processed_data,
    "output/porter_stemmed_comments.csv",
    "output/snowball_stemmed_comments.csv",
)
