
import argparse
def main(args):
    #model = load_model(args)
    #print(args.data_path)
    print(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--output_file_path",
        type=str,
        default=None,
        required=True
    )
    # parser.add_argument(
    #     "--cmlm_model_path",
    #     type=str,
    #     required=True,
    # )
    # parser.add_argument(
    #     "--data_name_or_path",
    #     type=str,
    #     required=True,
    # )
    # parser.add_argument(
    #     "--mlm_path",
    #     type=str,
    #     required=True,
    # )
    # parser.add_argument(
    #     "--knn_model_path",
    #     type=str,
    #     required=True,
    # )
    
    args = parser.parse_args()
    main(args)