import argparse

from model import CLIPModel

def main():
    parser = argparse.ArgumentParser(description="Test")

    parser.add_argument("-i", "--input_dir", type=str, help="Input directory path")
    parser.add_argument("-n", "--name", type=str, help="Input dataset name")

    args = parser.parse_args()

    if args.input_dir is None or args.name is None:
        raise ValueError(
            "Input must have a valid input directory and a dataset name"
        )

    model = CLIPModel()

    model.preprocess_images(args.input_dir, args.name)
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()