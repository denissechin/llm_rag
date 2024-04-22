# LLM with Retrieval-augmented generation
This repository contains a simple in-terminal application that utilizes RAG for news question answering.

Default setup that I used:

* Dataset: lenta-ru-news from [here](https://github.com/yutkin/Lenta.Ru-News-Dataset)
* LLM: Saiga_llama3_8b deployed on llama.cpp from [here](https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf)
* Vector db: FAISS

# Usage

* Clone repository 
```
git clone https://github.com/denissechin/llm_rag.git
cd llm_rag
```
* If you want to use GPU offloading for LLM inference, install llamacpp with CUBLAS:
```
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```
* Install all libraries 
```
pip install -U pip
pip install -r requirements.txt
```
* Either download .csv file from lenta-ru-news repository and create database with your desired number of row news
```
python scripts/create_db.py --csv_path YOUR_PATH 
```
* Or download my prebuilt database for 200k rows with chunk_size=512 from here: [gdrive](https://drive.google.com/file/d/1JlEwEjkHTpMcDgjAMizutUkQjHQqZkVF/view?usp=sharing)
* Either download your desired LLM model in .gguf format and run
```
python scripts/rag_question_answering.py --db DB_PATH --model_path MODEL_PATH
```
* or run script without ```--model_path``` argument, default model will be downloaded to ./models/ folder:
```
python scripts/rag_question_answering.py --db DB_PATH
```