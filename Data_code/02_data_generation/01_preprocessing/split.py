import os
import argparse
from pdf2image import convert_from_path


def pdf_to_images_for_folder(input_dir: str, output_dir: str, dpi: int = 300):
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(input_dir, filename)
        pdf_name = os.path.splitext(filename)[0]
        pdf_output_dir = os.path.join(output_dir, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)

        try:
            pages = convert_from_path(pdf_path, dpi=dpi)
            for idx, page in enumerate(pages, start=1):
                img_path = os.path.join(pdf_output_dir, f"{idx}.png")
                page.save(img_path, "PNG")
        except Exception as e:
            raise RuntimeError(f"Failed to process {pdf_path}: {e}") from e


def main():
    parser = argparse.ArgumentParser(
        description="Split all PDF files in a folder into per-page PNG images."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory for generated images.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Output image DPI (higher = clearer, but slower/larger). Default: 400.",
    )

    args = parser.parse_args()
    pdf_to_images_for_folder(args.input_dir, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()