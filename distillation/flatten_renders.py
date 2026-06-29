import argparse
import glob
import os

from PIL import Image, UnidentifiedImageError


def flatten_directory(images_dir: str) -> int:
    flattened = 0
    skipped = 0
    for path in sorted(glob.glob(f"{images_dir}/*.png")):
        try:
            image = Image.open(path)
            image.load()
        except (UnidentifiedImageError, OSError):
            os.remove(path)
            skipped += 1
            continue
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            Image.alpha_composite(background, image).convert("RGB").save(path)
            flattened += 1
    if skipped:
        print(f"removed {skipped} unreadable images in {images_dir}")
    return flattened


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten transparent MuseScore3 PNG renders onto a white background."
    )
    parser.add_argument("--images-dir", required=True)
    args = parser.parse_args()
    count = flatten_directory(args.images_dir)
    print(f"flattened {count} images in {args.images_dir}")


if __name__ == "__main__":
    main()
