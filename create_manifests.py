import argparse

from utils.create_dataset import convert_hf_dataset_nemo


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face dataset to NeMo format."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the Hugging Face dataset (e.g., 'squad', 'imdb', 'glue').",
    )
    args = parser.parse_args()

    convert_hf_dataset_nemo(dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
