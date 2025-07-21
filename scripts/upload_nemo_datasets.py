import os
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, HfFolder, Repository
from tqdm import tqdm

# Config
CACHE_DIR = Path.home() / ".cache" / "datasets"
OWNER = "hsekhalilian"
CHUNK_SIZE_MB = 500
REPO_NAME = "fleurs_nemo_02"


def get_dataset_dir():
    for path in CACHE_DIR.glob("*___*"):
        if (
            path.is_dir()
            and (path / "audio_files").exists()
            and (path / "manifests").exists()
        ):
            return path
    raise FileNotFoundError("No matching dataset directory found.")


def split_audio_into_chunks(audio_dir, output_dir, max_chunk_size_mb=500):
    files = sorted(audio_dir.glob("**/*"), key=lambda f: f.stat().st_size)
    chunk_index = 0
    current_size = 0
    tar = None

    for file in tqdm(files, desc="Compressing audio files"):
        if not file.is_file():
            continue

        size_mb = file.stat().st_size / (1024 * 1024)
        if tar is None or current_size + size_mb > max_chunk_size_mb:
            if tar:
                tar.close()
            chunk_path = output_dir / f"chunk_{chunk_index:03}.tar.gz"
            tar = tarfile.open(chunk_path, "w:gz")
            chunk_index += 1
            current_size = 0
        arcname = file.relative_to(audio_dir)
        tar.add(file, arcname=arcname)
        current_size += size_mb

    if tar:
        tar.close()


def create_or_clone_repo(repo_name, local_dir):
    api = HfApi()
    full_repo = f"{OWNER}/{repo_name}"
    token = HfFolder.get_token()
    try:
        api.repo_info(full_repo)
    except Exception:
        print(f"Creating repo {full_repo}...")
        api.create_repo(repo_name, repo_type="dataset", token=token, private=False)

    repo = Repository(
        local_dir=str(local_dir), clone_from=full_repo, repo_type="dataset", token=token
    )
    return repo


def main():
    dataset_dir = get_dataset_dir()
    audio_dir = dataset_dir / "audio_files"
    print(f"Found dataset at: {dataset_dir}")

    # Prepare local repo directory
    local_repo_dir = Path.cwd() / REPO_NAME
    chunks_dir = local_repo_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Create or clone repo
    repo = create_or_clone_repo(REPO_NAME, local_repo_dir)

    # Compress audio files into chunks
    split_audio_into_chunks(audio_dir, chunks_dir, CHUNK_SIZE_MB)

    # Push to Hugging Face Hub
    repo.push_to_hub(commit_message="Upload compressed audio chunks")


if __name__ == "__main__":
    main()
