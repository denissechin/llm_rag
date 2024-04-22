import argparse
import os
import time

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

MONTH_MAP = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}


def russify_date(date: str) -> str:
    """Creates a written version of date by changing yyyy/mm/dd to "dd of {month} of yyyy year"

    Args:
        date (str): string of format "yyyy/mm/dd"

    Returns:
        str: string of format "dd of {month} of yyyy year"
    """
    year, month, day = date.split("/")
    russified_date = f"{day} {MONTH_MAP[int(month)]} {year} года"
    return russified_date


if __name__ == "__main__":
    """
    Script to initialize and locally save FAISS vector database from lenta-ru-news dataset
    Dataset can be truncated to some number of rows from the start by passing --dataset_length argument
    Initialization of DB for 200k rows ran for somewhat around 30 min on RTX 3050 Ti and i7-11800H
    Mean length of texts is ~1350 chars, std 550 chars, so on average news text splits in three with 512 chunk size
    Also this script adds a date prefix to every chunk for better date-content alignment
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, help="Path to Lenta-ru-news dataset", required=True
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path where to save FAISS DB",
        default="./db/",
    )
    parser.add_argument(
        "--dataset_length",
        type=int,
        help="Number of rows from dataset's start to process and save to DB",
        default=200_000,
    )
    parser.add_argument(
        "--chunk_size", type=int, help="Number of characters in each chunk", default=512
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="(Optional) Path to folder where to save cache of embedding model",
        default="./models/",
    )

    args = parser.parse_args()

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        cache_folder=args.cache_dir,
        model_kwargs={"device": "cuda"},
    )

    df = pd.read_csv(args.csv_path)
    df = df.dropna()
    df = df.drop(["url", "topic", "tags"], axis=1)
    df = df[: args.dataset_length]

    chunk_size = 512
    docs_by_dates = {}

    # aggregate so that news for each date will be stored in a list
    aggregated = df.groupby("date").text.agg(lambda x: list(x))
    for date, texts in zip(aggregated.index, aggregated):
        docs_by_dates[date] = texts

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 7),
        strip_whitespace=False,
    )

    full_docs = []
    for date, texts in docs_by_dates.items():
        for text in texts:
            docs = text_splitter.split_text(text)
            docs = [
                " ".join((russify_date(date), doc)) for doc in docs
            ]  # add date as prefix
            full_docs.extend(docs)

    init_time = time.time()
    db = FAISS.from_texts(
        full_docs, embeddings, distance_strategy=DistanceStrategy.COSINE
    )

    save_path = os.path.join(args.save_path, f"faiss_{args.dataset_length // 1000}k")
    db.save_local(save_path)

    total_time = (time.time() - init_time) / 60
    print(
        f"\nDB for {args.dataset_length // 1000}k rows initialized in {total_time:.2f} minutes and saved into {os.path.abspath(save_path)}"
    )
