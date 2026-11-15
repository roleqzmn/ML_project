"""Index the supplied audio into source-disjoint splits and ratio batches.

Audio remains in place. Existing smart_chunk_wav manifests are preferred.
Without them, long negatives become virtual nonoverlapping two-second windows.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import wave
from collections import Counter, defaultdict
from pathlib import Path

from batching import RatioBatchSampler

FIELDS = ["path", "label", "split", "speaker_id", "group_id",
          "start_frame", "num_frames", "sample_rate", "source_stratum"]
SPLITS = ("train", "val", "test")


def audio_info(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getframerate(), audio.getnframes()


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def allowed_assignments(groups, rng):
    """Use whole recordings; with only two, retain train + test and flag val."""
    by_speaker = defaultdict(list)
    for group, rows in groups.items():
        by_speaker[rows[0]["speaker_id"]].append(group)
    assignments, warnings = {}, []
    for speaker, ids in sorted(by_speaker.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        if len(ids) < 2:
            raise ValueError(f"{speaker}: at least two original recordings required")
        if len(ids) == 2:
            ids.sort(key=lambda group: len(groups[group]), reverse=True)
            assignments[ids[0]], assignments[ids[1]] = "train", "test"
            warnings.append(f"{speaker}: only two source recordings; no samples in validation.")
            continue
        total = sum(len(groups[group]) for group in ids)
        # Leave at least one complete source for each held-out split.
        # Exhaustive search is inexpensive for the small number of sessions here.
        choices = []
        for val, test in itertools.permutations(ids, 2):
            nv, nt = len(groups[val]), len(groups[test])
            score = ((nv / total - .2) ** 2 + (nt / total - .2) ** 2
                     + ((total - nv - nt) / total - .6) ** 2)
            choices.append((score, val, test))
        _, val, test = min(choices, key=lambda item: item[0])
        for group in ids:
            assignments[group] = "val" if group == val else "test" if group == test else "train"
    return assignments, warnings


def allocate_negative_groups(groups, fractions, rng):
    """Stratify by source folder/language, balancing numbers of windows."""
    strata = defaultdict(list)
    for group, rows in groups.items():
        strata[rows[0].get("source_stratum") or str(Path(rows[0]["path"]).parent)].append(group)
    assignments = {}
    for _, ids in sorted(strata.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        ids.sort(key=lambda group: len(groups[group]), reverse=True)
        total = sum(len(groups[group]) for group in ids)
        target = {s: fractions[s] * total for s in SPLITS}
        counts = dict.fromkeys(SPLITS, 0)
        for group in ids:
            weight = len(groups[group])
            # Minimize the change in weighted squared deviation from the targets.
            split = min(SPLITS, key=lambda s:
                        (2 * (counts[s] - target[s]) * weight + weight ** 2) / max(target[s], 1))
            assignments[group] = split
            counts[split] += weight
    return assignments


def prepare(args):
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    if args.seconds < 1 or not 0 < args.min_seconds <= args.seconds:
        raise ValueError("Require seconds >= 1 and 0 < min_seconds <= seconds")
    rng = random.Random(args.seed)
    registry = output / "additional_allowed_sources.json"
    extra_sources = json.loads(registry.read_text(encoding="utf-8")) if registry.is_file() else []
    for manifest, speaker, split in args.extra_allowed or []:
        manifest = str(Path(manifest).resolve())
        entry = dict(manifest=manifest, speaker_id=speaker, split=split)
        existing = [item for item in extra_sources if str(Path(item["manifest"]).resolve()) == manifest]
        if existing and existing[0] != entry:
            raise ValueError(f"Conflicting additional source metadata: {manifest}")
        if not existing:
            extra_sources.append(entry)
    for entry in extra_sources:
        if entry["split"] not in SPLITS or not entry["speaker_id"].strip():
            raise ValueError("Additional allowed sources require a speaker and train/val/test split")
    extra_paths = {Path(entry["manifest"]).resolve() for entry in extra_sources}
    candidates = sorted((data_root / "allowed").rglob("manifest.csv"))
    candidates = [p for p in candidates if p.resolve() not in extra_paths]
    if len(candidates) != 1:
        raise ValueError("Expected one base allowed manifest; register additions with --extra-allowed")
    source_manifest = candidates[0]
    positive_groups = defaultdict(list)
    indexed_paths = set()
    forced_assignments = {}
    inputs = [dict(manifest=str(source_manifest), speaker_id=None, split=None)] + extra_sources
    for entry in inputs:
        manifest_path = Path(entry["manifest"]).resolve()
        extra_tag = hashlib.sha256(str(manifest_path).encode("utf-8")).hexdigest()[:12]
        with manifest_path.open(encoding="utf-8-sig", newline="") as f:
            for source in csv.DictReader(f):
                if source["label"] != "1":
                    raise ValueError(f"Allowed manifest has a label other than 1: {manifest_path}")
                path = (manifest_path.parent / source["chunk_file"]).resolve()
                if not path.is_relative_to(data_root / "allowed"):
                    raise ValueError(f"Allowed chunk outside allowed directory: {path}")
                if path in indexed_paths:
                    raise ValueError(f"Repeated allowed chunk: {path}")
                indexed_paths.add(path)
                speaker = entry["speaker_id"] or Path(source["chunk_file"]).parts[0]
                prefix = f"extra_{extra_tag}/" if entry["split"] else ""
                group = f"allowed/{speaker}/{prefix}{source['source_group']}"
                sr, frames = audio_info(path)
                if frames < 1:
                    raise ValueError(f"Empty allowed recording: {path}")
                positive_groups[group].append(dict(path=str(path), label=1, split="",
                    speaker_id=speaker, group_id=group, start_frame=0,
                    num_frames=frames, sample_rate=sr))
                if entry["split"]:
                    forced_assignments[group] = entry["split"]
    all_positive_files = {p.resolve() for p in (data_root / "allowed").rglob("*.wav")}
    if indexed_paths != all_positive_files:
        raise ValueError("Allowed source manifest does not match the available WAV files")
    positive_assignments, warnings = allowed_assignments(positive_groups, rng)
    previous_assignments = {}
    previous_file = output / "source_assignments.csv"
    if previous_file.is_file() and not args.reshuffle_splits:
        with previous_file.open(encoding="utf-8-sig", newline="") as f:
            previous_assignments = {r["group_id"]: r["split"] for r in csv.DictReader(f)}
        positive_assignments.update({g: previous_assignments[g] for g in positive_groups
                                     if g in previous_assignments})
    for group, split in forced_assignments.items():
        if group in previous_assignments and previous_assignments[group] != split:
            raise ValueError(f"Additional recording conflicts with its existing split: {group}")
        positive_assignments[group] = split
    positive_counts = Counter()
    for group, rows in positive_groups.items():
        for row in rows:
            row["split"] = positive_assignments[group]
            positive_counts[row["split"]] += 1
    total_positive = sum(positive_counts.values())
    fractions = {s: positive_counts[s] / total_positive for s in SPLITS}

    negative_groups, skipped = defaultdict(list), []
    negative_root = data_root / "not_allowed"
    chunk_manifests = sorted(negative_root.rglob("manifest.csv"))
    chunking_summary = None
    if chunk_manifests:
        if len(chunk_manifests) != 1:
            raise ValueError("Expected exactly one not_allowed chunk manifest")
        chunk_manifest = chunk_manifests[0]
        run = json.loads((chunk_manifest.parent / "run.json").read_text(encoding="utf-8"))
        chunking_summary = run["summary"]
        if chunking_summary["errors"]:
            raise ValueError("Audio chunking reported errors; inspect skipped.csv first")
        negative_source_count = chunking_summary["files"]
        indexed_chunks = set()
        with chunk_manifest.open(encoding="utf-8-sig", newline="") as f:
            for index, source in enumerate(csv.DictReader(f), 1):
                if source["label"] != "0":
                    raise ValueError("Negative chunk manifest must have label 0")
                path = (chunk_manifest.parent / source["chunk_file"]).resolve()
                if not path.is_relative_to(negative_root) or path in indexed_chunks:
                    raise ValueError(f"Invalid or duplicate negative chunk: {path}")
                indexed_chunks.add(path)
                sr, frames = audio_info(path)
                if frames < 1:
                    raise ValueError(f"Empty negative chunk: {path}")
                group = "not_allowed/" + source["source_group"]
                negative_groups[group].append(dict(path=str(path), label=0, split="",
                    speaker_id="", group_id=group, start_frame=0,
                    num_frames=frames, sample_rate=sr,
                    source_stratum=Path(source["source_file"]).parent.as_posix()))
                if index % 5000 == 0:
                    print(f"Indexed speech chunks: {index}", flush=True)
        if len(indexed_chunks) != chunking_summary["chunks"]:
            raise ValueError("Chunk manifest count does not match run.json")
    else:
        negative_files = sorted(negative_root.rglob("*.wav"))
        negative_source_count = len(negative_files)
        if not negative_files:
            raise ValueError("No not_allowed WAV files found")
        for index, path in enumerate(negative_files, 1):
            sr, frames = audio_info(path)
            step, minimum = round(args.seconds * sr), math.ceil(args.min_seconds * sr)
            group = "not_allowed/" + path.relative_to(negative_root).as_posix()
            for start in range(0, frames, step):
                length = min(step, frames - start)
                if length < minimum:
                    skipped.append(dict(path=str(path), start_frame=start, num_frames=length,
                                        sample_rate=sr, reason="shorter_than_min_seconds"))
                    continue
                negative_groups[group].append(dict(path=str(path), label=0, split="",
                    speaker_id="", group_id=group, start_frame=start,
                    num_frames=length, sample_rate=sr,
                    source_stratum=path.parent.relative_to(negative_root).as_posix()))
            if index % 1000 == 0:
                print(f"Indexed negative recordings: {index}/{len(negative_files)}", flush=True)
    warnings.append("Unknown speaker IDs unavailable: negative splits are source-file-disjoint, not verified speaker-disjoint.")
    warnings.append("Allowed group IDs identify source recordings; separate recording sessions are not verified.")
    warnings.append("This indexing step does not normalize audio or detect duplicate audio content.")
    if not chunking_summary:
        warnings.append("No speech detection was performed; negative segments use fixed time windows.")
    negative_assignments = allocate_negative_groups(negative_groups, fractions, rng)
    negative_assignments.update({g: previous_assignments[g] for g in negative_groups
                                 if g in previous_assignments})
    for group, rows in negative_groups.items():
        for row in rows:
            row["split"] = negative_assignments[group]
    rows = [row for groups in (positive_groups, negative_groups)
            for group in sorted(groups) for row in groups[group]]
    if len({(r["path"], r["start_frame"], r["num_frames"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate segments detected")
    path_split, group_split = {}, {}
    for row in rows:
        for mapping, key in [(path_split, row["path"]), (group_split, row["group_id"])]:
            if mapping.setdefault(key, row["split"]) != row["split"]:
                raise ValueError("Source recording leaked across splits")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "manifest.csv", rows, FIELDS)
    counts, per_speaker = {}, {}
    for split in SPLITS:
        subset = [r for r in rows if r["split"] == split]
        write_csv(output / f"{split}.csv", subset, FIELDS)
        counts[split] = {"allowed": sum(r["label"] for r in subset),
                         "not_allowed": sum(r["label"] == 0 for r in subset), "total": len(subset)}
        per_speaker[split] = dict(sorted(Counter(r["speaker_id"] for r in subset if r["label"] == 1).items()))
    source_rows = [dict(group_id=group, split=group_rows[0]["split"],
                        label=group_rows[0]["label"], segments=len(group_rows),
                        speaker_id=group_rows[0]["speaker_id"])
                   for groups in (positive_groups, negative_groups)
                   for group, group_rows in sorted(groups.items())]
    write_csv(output / "source_assignments.csv", source_rows,
              ["group_id", "split", "label", "segments", "speaker_id"])
    write_csv(output / "skipped_segments.csv", skipped,
              ["path", "start_frame", "num_frames", "sample_rate", "reason"])
    train_rows = [r for r in rows if r["split"] == "train"]
    sampler = RatioBatchSampler([r["label"] for r in train_rows], args.batch_size,
                                args.seed, args.allowed_per_batch)
    batch_index = []
    for batch_id, indices in enumerate(sampler):
        for position, sample in enumerate(indices):
            batch_index.append(dict(batch_id=batch_id, position=position,
                                    train_row=sample, label=train_rows[sample]["label"]))
    write_csv(output / "train_batches_epoch_1.csv", batch_index,
              ["batch_id", "position", "train_row", "label"])
    summary = dict(data_root=str(data_root), seed=args.seed,
                   additional_allowed_sources=extra_sources,
                   existing_source_splits_preserved=bool(previous_assignments),
                   negative_index_mode="smart_speech_chunks" if chunking_summary else "virtual_windows",
                   chunking_summary=chunking_summary,
                   window_seconds=None if chunking_summary else args.seconds,
                   minimum_window_seconds=args.min_seconds, counts=counts,
                   allowed_per_speaker=per_speaker, warnings=warnings,
                   batch_size=args.batch_size, training_batches_per_epoch=len(sampler),
                   allowed_per_full_batch=sampler.allowed_per_batch,
                   not_allowed_per_full_batch=sampler.denied_per_batch,
                   recommended_pos_weight=sampler.denied_per_batch / sampler.allowed_per_batch,
                   samples_per_training_epoch=len(batch_index),
                   negative_source_files=negative_source_count,
                   negative_source_files_with_usable_segments=len(negative_groups),
                   skipped_short_segments=len(skipped),
                   audio_modified=False, unknown_speaker_disjointness_verified=False)
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    registry.write_text(json.dumps(extra_sources, indent=2), encoding="utf-8")
    lines = ["# Prepared data", "", "Audio stays in its original location.", "",
             "| Split | Allowed | Not allowed | Total |", "| --- | ---: | ---: | ---: |"]
    for split, count in counts.items():
        lines.append(f"| {split} | {count['allowed']} | {count['not_allowed']} | {count['total']} |")
    lines += ["", f"Training: {len(sampler)} batches per epoch, batch_size={args.batch_size}.",
              f"Full batches: {sampler.allowed_per_batch} allowed + {sampler.denied_per_batch} not_allowed.",
              "The last batch may be smaller. Allowed samples are drawn with replacement.",
              "Validation and test retain their original class proportions.", "",
              "train_batches_epoch_1.csv uses zero-based data row indices into train.csv (header excluded).",
              "Later epochs reshuffle samples automatically.", "", "## Limitations", ""]
    lines += [f"- {warning}" for warning in warnings]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Ready: {output / 'manifest.csv'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default="prepared_data")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--allowed-per-batch", type=int, default=None,
                   help="Allowed examples per full batch; default: batch_size/4 (ratio 1:3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra-allowed", nargs=3, action="append", metavar=("MANIFEST", "SPEAKER", "SPLIT"),
                   help="Register extra smart_chunk_wav output with a known speaker and fixed split")
    p.add_argument("--reshuffle-splits", action="store_true",
                   help="Reassign existing source groups; default: preserve existing assignments")
    prepare(p.parse_args())
