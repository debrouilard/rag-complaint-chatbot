import pandas as pd
import re


PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Savings account",
    "Money transfer"
]


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_data(input_file, output_file):

    df = pd.read_csv(input_file)

    print("Original Shape:", df.shape)

    df = df[
        df["Product"].isin(PRODUCTS)
    ]

    df = df.dropna(
        subset=["Consumer complaint narrative"]
    )

    df = df[
        df["Consumer complaint narrative"]
        .str.strip()
        .ne("")
    ]

    df["clean_narrative"] = (
        df["Consumer complaint narrative"]
        .apply(clean_text)
    )

    print("Filtered Shape:", df.shape)

    df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":

    preprocess_data(
        "data/raw/complaints.csv",
        "data/processed/filtered_complaints.csv"
    )