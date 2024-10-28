import argparse

from model import CLIPModel

def main():
    parser = argparse.ArgumentParser(description="Test")
    
    parser.add_argument("-q", "--query", type=str, help="Input text query")
    parser.add_argument("-n", "--name", type=str, help="Input dataset name")

    args = parser.parse_args()

    if args.query is None or args.name is None:
        raise ValueError(
            "Input must have a valid query and a preprocessed dataset name"
        )

    model = CLIPModel()
    model.search(args.query, args.name)

if __name__ == "__main__":
    main()