"""ISIC2017 dataset preparation entry point."""
from scripts.data_preparation.prepare_segmentation import main as _main

def main():
    _main("isic2017")

if __name__ == "__main__":
    main()
