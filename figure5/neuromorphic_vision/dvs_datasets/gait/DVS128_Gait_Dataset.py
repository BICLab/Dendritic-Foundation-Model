"""
@deprecated DVS128 Gait Dataset
"""
import argparse
from pathlib import Path

import numpy as np


def _convert_split(input_dir: Path, output_dir: Path, split: str) -> None:
    split_input = input_dir / split
    if not split_input.is_dir():
        raise FileNotFoundError(f"Missing DVS128 Gait split: {split_input}")

    split_output = output_dir / split
    split_output.mkdir(parents=True, exist_ok=True)

    samples = []
    targets = []
    for class_dir in sorted(path for path in split_input.iterdir() if path.is_dir()):
        label = int(class_dir.name)
        for sample_file in sorted(class_dir.glob("*.txt")):
            samples.append(np.loadtxt(sample_file, dtype=np.int64))
            targets.append(label)

    np.save(split_output / f"{split}_data.npy", np.asarray(samples))
    np.save(split_output / f"{split}_target.npy", np.asarray(targets))


def DVS128_Gait_txt_to_npy(path, save_path):
    input_dir = Path(path)
    output_dir = Path(save_path) / "npy"
    for split in ("train", "test"):
        _convert_split(input_dir, output_dir, split)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert DVS128 Gait text files to NumPy arrays.")
    parser.add_argument("--input-dir", required=True, help="directory containing train and test splits")
    parser.add_argument("--output-dir", required=True, help="destination root; an npy directory is created below it")
    args = parser.parse_args()
    DVS128_Gait_txt_to_npy(args.input_dir, args.output_dir)