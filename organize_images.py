import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "cancer.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy images into the folder paths listed in cancer.csv.")
    parser.add_argument(
        "source",
        type=Path,
        help="Folder containing the extracted image dataset",
    )
    args = parser.parse_args()
    source_dir = args.source.resolve()

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source folder does not exist: {source_dir}")

    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows or not {"images", "labels"}.issubset(rows[0]):
        raise ValueError("cancer.csv must contain images and labels columns.")

    image_index: dict[str, list[Path]] = defaultdict(list)
    for image_file in source_dir.rglob("*"):
        if image_file.is_file() and image_file.suffix.lower() == ".png":
            image_index[image_file.name].append(image_file)

    copies: list[tuple[Path, Path]] = []
    missing: list[str] = []
    ambiguous: list[str] = []

    for row in rows:
        relative_path = Path(row["images"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe image path in CSV: {row['images']}")

        matches = image_index.get(relative_path.name, [])
        if not matches:
            missing.append(row["images"])
            continue
        if len(matches) > 1:
            ambiguous.append(f"{row['images']}: {matches}")
            continue

        destination = (BASE_DIR / relative_path).resolve()
        if BASE_DIR.resolve() not in destination.parents:
            raise ValueError(f"Image path is outside the project folder: {row['images']}")
        copies.append((matches[0], destination))

    if missing or ambiguous:
        if missing:
            print(f"Missing images ({len(missing)}):")
            for image_path in missing[:20]:
                print(f"  {image_path}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
        if ambiguous:
            print(f"Ambiguous image filenames ({len(ambiguous)}):")
            for image_path in ambiguous[:20]:
                print(f"  {image_path}")
            if len(ambiguous) > 20:
                print(f"  ... and {len(ambiguous) - 20} more")
        raise FileNotFoundError(
            "No images were copied. Use the matching extracted dataset folder "
            "and resolve any duplicate filenames."
        )

    existing = [str(destination) for _, destination in copies if destination.exists()]
    if existing:
        raise FileExistsError(
            "No images were copied because destination files already exist:\n"
            + "\n".join(existing[:20])
        )

    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    print(f"Copied {len(copies)} images into: {BASE_DIR / 'data'}")


if __name__ == "__main__":
    main()
