import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


sample_rate = 16000
duration = 3
n_mels = 64
n_fft = 1024
hop_length = 256


def create_spectrogram(path):
    audio, sr = librosa.load(str(path), sr=sample_rate, mono=True)

    audio = librosa.util.fix_length(
        audio, size=sample_rate * duration
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    spectrogram = librosa.power_to_db(
        mel, ref=np.max, top_db=80
    )

    return spectrogram.astype(np.float32)


def save_plot(spec, path, title):
    plt.figure(figsize=(8, 4))

    librosa.display.specshow(
        spec,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        vmin=-80,
        vmax=0
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def process_chunks(folders, output, label):
    class_name = "allowed" if label == 1 else "not_allowed"
    results = []

    for folder in folders:
        with open(folder / "manifest.csv", encoding="utf-8") as file:
            chunks = list(csv.DictReader(file))

        for chunk in chunks:
            if chunk["label"] not in ("", str(label)):
                raise ValueError(
                    f"Wrong label in {chunk['chunk_file']}"
                )

            wav_path = folder / chunk["chunk_file"]

            spec = create_spectrogram(wav_path)

            npy_path = (
                Path("npy") / class_name /
                Path(chunk["chunk_file"]).with_suffix(".npy")
            )

            (output / npy_path).parent.mkdir(
                parents=True, exist_ok=True
            )

            np.save(output / npy_path, spec)

            row = dict(chunk)
            row["label"] = label
            row["npy_file"] = npy_path.as_posix()
            row["input_folder"] = folder.name

            results.append(row)

        print(f"{folder}: {len(chunks)} spectrograms created")

    examples = random.sample(results, min(3, len(results)))

    for i, row in enumerate(examples, start=1):
        spec = np.load(output / row["npy_file"])

        filename = Path(row["chunk_file"]).name
        length = float(row["duration_seconds"])

        title = f"{class_name} - {filename} - {length:.2f} s"

        save_plot(
            spec,
            output / "examples" / f"{class_name}_{i}.pdf",
            title
        )

    return results



def main():
    parser = argparse.ArgumentParser()

    project_dir = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("spectrograms")
    )

    parser.add_argument(
        "--allowed", type=Path, nargs="+", required=True
    )

    parser.add_argument(
        "--not-allowed", type=Path, nargs="+", required=True
    )

    args = parser.parse_args()

    output = (project_dir / args.output).resolve()

    for folders in (args.allowed, args.not_allowed):
        if len({folder.name for folder in folders}) != len(folders):
            parser.error(
                "Input folders of the same class need different names."
            )

        for folder in folders:
            if not (folder / "manifest.csv").is_file():
                parser.error(f"Missing manifest.csv in {folder}")

            if output == folder.resolve() or folder.resolve() in output.parents:
                parser.error(
                    "Output must be outside the chunk folders."
                )

    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        parser.error("Output folder must be new or empty.")

    output.mkdir(parents=True, exist_ok=True)
    (output / "examples").mkdir()

    random.seed(0)

    allowed = process_chunks(args.allowed, output, 1)
    not_allowed = process_chunks(args.not_allowed, output, 0)

    if not allowed or not not_allowed:
        raise ValueError("Both classes need at least one chunk.")

    results = allowed + not_allowed

    fields = list(
        dict.fromkeys(key for row in results for key in row)
    )

    with open(
        output / "spectrograms.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    settings = {
        "sample_rate": sample_rate,
        "duration": duration,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "top_db": 80,
        "shape": [64, 188],
        "seed": 0,
        "allowed": len(allowed),
        "not_allowed": len(not_allowed)
    }

    with open(
        output / "run.json", "w", encoding="utf-8"
    ) as file:
        json.dump(settings, file, indent=2)

    print(f"Allowed: {len(allowed)} spectrograms")
    print(f"Not allowed: {len(not_allowed)} spectrograms")
    print(f"Total: {len(results)} spectrograms")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()

