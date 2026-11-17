# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Store the git state of the repositories a training run depends on."""

from __future__ import annotations

import os
import pathlib


def store_code_state(log_dir: str, repos: list[str]) -> list[str]:
    """Write ``<log_dir>/git/<repo>.diff`` (commit, status and diff) for each repository.

    Mirrors ``rsl_rl.utils.logger.Logger._store_code_state`` as a standalone function.

    Returns:
        The paths of the written diff files.
    """
    try:
        import git
    except ImportError:
        print("GitPython not installed; cannot store code state.")
        return []

    git_log_dir = os.path.join(log_dir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths: list[str] = []
    for repository_file_path in repos:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
            tree = repo.head.commit.tree
            commit_hash = repo.head.commit.hexsha
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")
            continue

        repo_name = pathlib.Path(repo.working_dir).name
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
        if os.path.isfile(diff_file_name):
            continue

        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        with open(diff_file_name, "x", encoding="utf-8") as f:
            f.write(
                f"--- git commit ---\n{commit_hash}\n\n\n"
                f"--- git status ---\n{repo.git.status()} \n\n\n"
                f"--- git diff ---\n{repo.git.diff(tree)}"
            )
        file_paths.append(diff_file_name)
    return file_paths
