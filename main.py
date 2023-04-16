import os
from pathlib import Path

from git.repo.base import Repo
from langchain import OpenAI
from llama_index import (GPTSimpleVectorIndex, LLMPredictor, PromptHelper,
                         ServiceContext, download_loader)
from llama_index.node_parser import SimpleNodeParser

from src.context_loader import generate_files

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).parent.joinpath("test_project"))
)

TEMP_PATH = Path(os.environ.get("TEMP_PATH", Path(__file__).parent.joinpath("tmp")))
UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
loader = UnstructuredReader()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)


def prepare_temp_file(file):
    temp_file = TEMP_PATH.joinpath(f"{file.name} .txt")
    file_content = ""
    print("Processing file", file.name)
    with open(file, "r") as f:
        file_content = f.read()
    with open(temp_file, "w") as f:
        f.write(file_content)
    return temp_file


def extract_documents(filename_info, temp_file):
    file_docs = loader.load_data(file=temp_file, split_documents=False)
    for file_doc in file_docs:
        file_doc.extra_info = {"filename": filename_info}
    return file_docs


def create_llm_index(all_docs):
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(all_docs)
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0, model_name="text-davinci-003", openai_api_key=OPENAI_API_KEY
        )
    )
    max_input_size = 4096
    max_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, max_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    cur_index = GPTSimpleVectorIndex(nodes, service_context=service_context)
    # TODO: save index as repo name
    index_name = "index_test.json"
    cur_index.save_to_disk(str(TEMP_PATH.joinpath(index_name)))


def chat():
    index_name = "index_test.json"
    index = GPTSimpleVectorIndex.load_from_disk(str(TEMP_PATH.joinpath(index_name)))
    print("Chatting")
    while True:
        query = input("Query: ")
        if query == "exit":
            break
        response = index.query(query)
        print("Response:", response)


def main():
    all_docs = []
    repo = Repo(PROJECT_ROOT)
    files = generate_files(PROJECT_ROOT, repo)
    if not TEMP_PATH.exists():
        TEMP_PATH.mkdir()
    processed_files_path = TEMP_PATH.joinpath("processed_files")
    if not processed_files_path.exists():
        processed_files_path.mkdir()
    for file in files:
        try:
            temp_file = prepare_temp_file(file)
            filename_info = str(file.relative_to(PROJECT_ROOT))
            all_docs = all_docs + extract_documents(filename_info, temp_file)
        except Exception as e:
            print("error", e)

    create_llm_index(all_docs)
    chat()


if __name__ == "__main__":
    main()
