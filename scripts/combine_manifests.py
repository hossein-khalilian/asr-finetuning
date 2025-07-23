import argparse
import os


def combine_manifests(manifest_paths, output_dir):
    output_path = os.path.join(output_dir, "train_manifest.json")
    os.makedirs(output_dir, exist_ok=True)

    total_input_lines = 0

    with open(output_path, "w") as outfile:
        for path in manifest_paths:
            input_lines = 0
            with open(path, "r") as infile:
                for line in infile:
                    outfile.write(line)
                    input_lines += 1

            parent_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
            print(f"{parent_dir}: {input_lines} lines")
            total_input_lines += input_lines

    with open(output_path, "r") as f:
        total_output_lines = sum(1 for _ in f)

    print(f"Total output lines: {total_output_lines}")
    print(f"Combined manifest saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine multiple manifest JSON files into one."
    )
    parser.add_argument(
        "manifests", nargs="+", help="Paths to manifest JSON files to be combined."
    )
    parser.add_argument(
        "--output_dir",
        default="/home/user/.cache/asr-finetuning/datasets/combined",
        help="Directory to save the combined manifest file.",
    )
    args = parser.parse_args()

    combine_manifests(args.manifests, args.output_dir)
