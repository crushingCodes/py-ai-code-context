import fnmatch
from pathlib import Path

from git.repo.base import Repo


def is_ignored(repo: Repo, file_path: str) -> bool:
    # TODO: check if file is excluded by gitignore is working
    return ".git" in file_path or len(Repo.ignored(repo, file_path)) > 0


def generate_files(root: Path, repo: Repo, all_files=[]) -> list[Path]:
    paths = root.iterdir()
    working_tree_dir = repo.working_tree_dir
    if not working_tree_dir:
        raise ValueError("No working tree found")
    for path in paths:
        file_path_relative = str(path.relative_to(working_tree_dir))
        if path.is_dir() and not is_ignored(repo, file_path_relative):
            generate_files(path, repo, all_files)
        elif path.is_file() and not is_ignored(repo, file_path_relative):
            all_files.append(path)
    return all_files
