"""PH2 dataset preparation entry point."""
from scripts.data_preparation.prepare_segmentation import main as _main

def main():
    _main("ph2")

if __name__ == "__main__":
    main()
