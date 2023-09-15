import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Analysis")

    parser.add_argument(
        "--source_weights_path", required=True, help="Path to source weights file", type=str
    )
