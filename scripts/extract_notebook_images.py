import base64
import json
from pathlib import Path


NOTEBOOKS = [
    "analysis-data.ipynb",
    "data-processing.ipynb",
    "embedding.ipynb",
    "model-nlp-tk1.ipynb",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "assets" / "notebook-images"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for notebook_name in NOTEBOOKS:
        notebook_path = root / notebook_name
        if not notebook_path.exists():
            continue

        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        stem = notebook_path.stem.replace(" ", "-")
        image_index = 1

        for cell_index, cell in enumerate(notebook.get("cells", [])):
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                mime = None
                ext = None

                if "image/png" in data:
                    mime = "image/png"
                    ext = "png"
                elif "image/jpeg" in data:
                    mime = "image/jpeg"
                    ext = "jpg"

                if not mime:
                    continue

                encoded = data[mime]
                if isinstance(encoded, list):
                    encoded = "".join(encoded)

                file_name = f"{stem}-{image_index:02d}-cell-{cell_index}.{ext}"
                file_path = out_dir / file_name
                file_path.write_bytes(base64.b64decode(encoded))

                manifest.append(
                    {
                        "notebook": notebook_name,
                        "cell": cell_index,
                        "image": str(file_path.relative_to(root)).replace("\\", "/"),
                    }
                )
                image_index += 1

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Extracted {len(manifest)} images to {out_dir}")


if __name__ == "__main__":
    main()
