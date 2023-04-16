import os
from pathlib import Path

from git.repo.base import Repo
from llama_index import GPTSimpleVectorIndex, ServiceContext, download_loader

from src.context_loader import generate_files

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).parent.joinpath("test_project"))
)

TEMP_PATH = Path(os.environ.get("TEMP_PATH", Path(__file__).parent.joinpath("tmp")))
UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
loader = UnstructuredReader()


def prepare_temp_file(file, doc_set, all_docs, index_set, service_context):
    temp_file = TEMP_PATH.joinpath(f"{file.name} .txt")
    file_content = ""
    print("Processing file", file.name)
    with open(file, "r") as f:
        file_content = f.read()
    with open(temp_file, "w") as f:
        f.write(file_content)
    return temp_file


def ingest_data(
    doc_set: dict,
    all_docs: list,
    index_set: dict,
    service_context,
    temp_file,
    filename_info: str,
):
    file_docs = loader.load_data(file=temp_file, split_documents=False)
    for file_doc in file_docs:
        file_doc.extra_info = {"filename": filename_info}
    doc_set[filename_info] = file_docs
    all_docs.extend(file_docs)

    cur_index = GPTSimpleVectorIndex.from_documents(
        doc_set[filename_info], service_context=service_context
    )
    # remove slashes from filename_info
    filename_info_index = filename_info.replace("/", "_")
    index_name = f"index_{filename_info_index}.json"
    cur_index.save_to_disk(str(TEMP_PATH.joinpath(index_name)))
    index_set[filename_info] = cur_index


def main():
    doc_set = {}
    all_docs = []
    index_set = {}
    service_context = ServiceContext.from_defaults(chunk_size_limit=512)
    repo = Repo(PROJECT_ROOT)
    files = generate_files(PROJECT_ROOT, repo)
    if not TEMP_PATH.exists():
        TEMP_PATH.mkdir()
    processed_files_path = TEMP_PATH.joinpath("processed_files")
    if not processed_files_path.exists():
        processed_files_path.mkdir()
    for file in files:
        try:
            temp_file = prepare_temp_file(
                file, doc_set, all_docs, index_set, service_context
            )
            filename_info = str(file.relative_to(PROJECT_ROOT))
            ingest_data(
                doc_set, all_docs, index_set, service_context, temp_file, filename_info
            )
        except Exception as e:
            print("error", e)


if __name__ == "__main__":
    main()
