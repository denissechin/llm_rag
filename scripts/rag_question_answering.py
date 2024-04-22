import argparse
import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from model import LLMForRAG

if __name__ == "__main__":
    """
    In-terminal app for RAG LLM QA on Lenta news dataset
    Tested usage:
    Model: Saiga2 7b / Saiga2 13b / Saiga3 8b Q4_K quantized models
    Vector Database: FAISS DB on 200k rows from lenta-ru-news
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to LLM weights file in .gguf format",
        default=None,
    )
    parser.add_argument(
        "--db", type=str, help="Path to FAISS vector database directory", required=True
    )
    parser.add_argument(
        "--topk", type=int, help="Number of chunks retrieved from database", default=5
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="(Optional) Path to folder where to save cache of embedding model",
        default="./models/",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print run times of database and LLM",
        dest="verbose",
        default=False,
    )

    args = parser.parse_args()

    ## SET UP MODEL
    print("Setting up model" + "." * 5)

    n_gpu_layers = 20  # you will have to adapt this variable to your model and GPU, this works for 3050ti and saiga3-8b
    n_threads = (
        8  # how many threads will be used for CPU processing, None for all threads
    )
    n_batch = 256
    n_ctx = 2048  # 2048 seems to be optimal as default, you may need to have to adapt this for bigger/lesser chunk size
    max_tokens = 256  # max generated tokens
    temperature = 0.5  # model temperature, less temperature -> less creative model
    llama_verbose = False

    llm = LLMForRAG(
        model_path=args.model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_threads=n_threads,
        verbose=llama_verbose,
    )

    SYSTEM_PROMPT = "Используя информацию из контекста, дай конкретный и детальный ответ на заданный вопрос. Думай шаг за шагом."
    USER_TEMPLATE = """
            Контекст: {context}
            Вопрос: {question}
            Ответ:"""

    print("Model set up successfully!")

    ## SET UP DB

    print("\nSetting up FAISS DB" + "." * 5)

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        cache_folder=args.cache_dir,
        model_kwargs={"device": "cpu"},
    )
    db = FAISS.load_local(args.db, embeddings, allow_dangerous_deserialization=True)

    print("FAISS DB set up successfully!")

    ## MAIN CYCLE

    while True:
        question = input("\nEnter your question...\n")

        db_time = time.time()

        retrieved_docs = db.similarity_search(question, k=5)
        context = "".join(doc.page_content + "\n\n" for doc in retrieved_docs)
        if args.verbose:
            print(f"DB similarity search in {time.time() - db_time :.2f} seconds")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(context=context, question=question),
            },
        ]

        llm_time = time.time()

        result = llm.invoke(messages)
        if args.verbose:
            print(f"LLM inference in {time.time() - db_time :.2f} seconds")

        print(result)
