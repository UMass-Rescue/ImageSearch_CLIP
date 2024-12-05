import argparse

from util.metrics import evaluate_image_search


def main():
    parser = argparse.ArgumentParser(description="Test")

    parser.add_argument("-i", "--input_dir", type=str, help="Input directory path")
    parser.add_argument("-n", "--name", type=str, help="Input dataset name")

    args = parser.parse_args()

    if args.name is None or args.input_dir is None:
        raise ValueError("Input must have a valid dataset name and directory")

    metrics_summary = evaluate_image_search(args.input_dir, args.name)
    print("Metrics Summary:", metrics_summary)


if __name__ == "__main__":
    main()
