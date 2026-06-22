import pandas as pd
from sklearn.model_selection import train_test_split
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_text(
        str(text)
    )