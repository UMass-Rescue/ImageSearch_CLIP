import argparse

from model.model import CLIPModel

def main():
    parser = argparse.ArgumentParser(description="Test")
    
    parser.add_argument("-i", "--image", type=str, help="Input image path")
    parser.add_argument("-q", "--query", type=str, help="Input text query")
    parser.add_argument("-n", "--name", type=str, help="Input dataset name")
    parser.add_argument("-k", "--num_results", type=int, help="Input num results")

    args = parser.parse_args()

    if args.name is None:
        raise ValueError(
            "Input must have a valid query and a preprocessed dataset name"
        )
    if args.query is None and args.image is None:
        raise ValueError(
            "Input must have a valid text query or image query path"
        )

    model = CLIPModel()
    if args.query is not None:
        model.search_by_text(args.query, args.name, args.num_results)
    else:
        model.search_by_image(args.image, args.name, args.num_results)

if __name__ == "__main__":
    main()
    