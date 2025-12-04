#src/utils/conf_matrix_one_pic.py
'''Plot Confusion Matrix for One-vs-Rest Multilabel Classifier'''

import os
import glob
import math
from PIL import Image

# Path to confusion matrix images (relative to THIS script)
IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         "../../data/imdb_arh_multilabel_conf"))
PATTERN = "ml_nb_conf_*.png"
OUTPUT_PATH = os.path.join(IMAGE_DIR, "ml_nb_all_confs.png")


def get_genre_from_filename(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    prefix = "ml_nb_conf_"
    return name[len(prefix):] if name.startswith(prefix) else name


def main():
    print("Looking in:", IMAGE_DIR)

    pattern_path = os.path.join(IMAGE_DIR, PATTERN)
    image_paths = glob.glob(pattern_path)

    if not image_paths:
        print(f"No files found matching: {pattern_path}")
        return

    # Sort alphabetically by genre name
    image_paths = sorted(image_paths, key=get_genre_from_filename)

    print("Found images in this order:")
    for p in image_paths:
        print("  ", os.path.basename(p))

    # Open images
    images = [Image.open(p) for p in image_paths]
    n = len(images)

    # Assume all images same size (as they come from the same plotting code)
    w, h = images[0].size

    # Compute grid size: square-ish
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    print(f"Arranging {n} images as {rows} rows x {cols} cols")

    # Create canvas for grid
    grid_width = cols * w
    grid_height = rows * h
    combined = Image.new("RGB", (grid_width, grid_height), "white")

    # Paste images row by row
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * w
        y = row * h
        combined.paste(img, (x, y))

    combined.save(OUTPUT_PATH)
    print(f"\n[OK] Saved grid image to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
