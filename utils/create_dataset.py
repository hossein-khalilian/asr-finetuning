import json
from pathlib import Path
from typing import Dict

import torch
import torchaudio
from datasets import DatasetDict, load_dataset
from text_normalizer.persian_normalizer import persian_normalizer_no_punc
from tqdm import tqdm

TARGET_SAMPLING_RATE = 16000


def resample_audio(
    waveform: torch.Tensor, orig_sr: int, target_sr: int
) -> torch.Tensor:
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=target_sr
        )
        return resampler(waveform)
    return waveform


def save_audio(waveform: torch.Tensor, sr: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, sr)


def process_sample(sample: Dict, output_dir: Path) -> Dict:
    text = (
        sample.get("text")
        or sample.get("sentence")
        or sample.get("raw_transcription")
        or sample.get("transcription")
        or sample.get("normalized_transcription")
    )
    text = persian_normalizer_no_punc(text)

    audio_info = sample["audio"]
    audio_filename = Path(audio_info["path"]).name
    output_audio_path = output_dir / "audio_files" / audio_filename

    waveform = torch.tensor(audio_info["array"]).unsqueeze(0)
    sr = audio_info["sampling_rate"]

    if sr != TARGET_SAMPLING_RATE:
        waveform = resample_audio(waveform, sr, TARGET_SAMPLING_RATE)

    save_audio(waveform, TARGET_SAMPLING_RATE, output_audio_path)

    duration = waveform.shape[1] / TARGET_SAMPLING_RATE

    return {
        "audio_filepath": str(output_audio_path.resolve()),
        "duration": duration,
        "text": text,
    }


def create_nemo_dataset(config: Dict) -> Path:
    dataset_name = config["dataset"]
    split = config.get("split", "test")
    sample_size = config.get("sample_size")

    output_dir = Path.home() / ".cache" / "datasets" / dataset_name.replace("/", "___")
    manifest_path = output_dir / f"{split}_manifest.json"

    # Load dataset to determine expected length
    dataset = load_dataset(dataset_name, split=split)
    if sample_size:
        dataset = dataset.select(range(sample_size))

    expected_len = len(dataset)

    # If the manifest exists and has the expected number of lines, skip processing
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            actual_len = sum(1 for _ in f)
        if actual_len == expected_len:
            print(
                f"Dataset already processed with {actual_len} entries: {manifest_path}"
            )
            return manifest_path
        else:
            print(
                f"Manifest found but length mismatch ({actual_len} != {expected_len}), regenerating..."
            )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w") as fout:
        for sample in tqdm(dataset, desc=f"Processing {split} split"):
            metadata = process_sample(sample, output_dir)
            fout.write(json.dumps(metadata) + "\n")

    return manifest_path


def convert_hf_dataset_nemo(
    dataset_name,
    output_dir=Path.home() / ".cache" / "asr-finetuning",
    split=None,
) -> Path:

    if isinstance(split, str):
        dataset = DatasetDict({split: load_dataset(dataset_name, split=split)})
        splits = [split]
    else:
        dataset = load_dataset(dataset_name)
        splits = list(dataset.keys())

    output_dir = output_dir / "datasets" / dataset_name.replace("/", "___")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        manifest_path = output_dir / "manifests" / f"{split}_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with manifest_path.open("w") as fout:
            for sample in tqdm(dataset[split], desc=f"Processing {split} split"):
                metadata = process_sample(sample, output_dir)
                if metadata["text"]:
                    fout.write(json.dumps(metadata) + "\n")

    return output_dir
