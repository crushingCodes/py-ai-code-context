import os
from pathlib import Path

from git.repo.base import Repo
from llama_index import GPTSimpleVectorIndex, ServiceContext, download_loader

from src.context_loader import generate_files

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).parent.joinpath("test_project"))
)
print(PROJECT_ROOT)

UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
loader = UnstructuredReader()
doc_set = {}
all_docs = []

if __name__ == "__main__":
    repo = Repo(PROJECT_ROOT)
    files = generate_files(PROJECT_ROOT, repo)
    temp_path = Path(__file__).parent.joinpath("tmp")
    if not temp_path.exists():
        temp_path.mkdir()
    processed_files_path = temp_path.joinpath("processed_files")
    if not processed_files_path.exists():
        processed_files_path.mkdir()
    for file in files:
        temp_file = temp_path.joinpath(f"{file.name} .txt")
        file_content = ""
        print("Processing file", file.name)
        try:
            with open(file, "r") as f:
                file_content = f.read()
            with open(temp_file, "w") as f:
                f.write(file_content)
            file_docs = loader.load_data(file=temp_file, split_documents=False)
            filename_info = file.name
            for file_doc in file_docs:
                file_doc.extra_info = {"filename": filename_info}
            all_docs.extend(file_docs)
        except Exception as e:
            print("error", e)
            continue

    print(len(all_docs))
