#!/usr/bin/env python3
"""Fit the two local judges for the AgServe quality-aware client adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.agserve_qac import (  # noqa: E402
    AGSERVE_QJ_MAX_LENGTH,
    AGSERVE_RJ_MAX_LENGTH,
    AGSERVE_UPSTREAM_REVISION,
)
from papc.paths import DATA_ROOT

DEFAULT_SHARED = Path(
    f"{DATA_ROOT}/papc/shared_endpoint_dev16_calibration_v1"
)
DEFAULT_STATE_RJ = Path(
    f"{DATA_ROOT}/papc/stepwise_state_bench_pilot_v1/"
    "datasets_v2/stuck_dataset.json"
)
DEFAULT_AGENT_RJ = Path(
    f"{DATA_ROOT}/papc/stepwise_agent_diff_pilot_v3/"
    "datasets/stuck_dataset.json"
)
DEFAULT_RETAIL_PAIRED = Path(
    f"{DATA_ROOT}/papc/"
    "external_router_live_publication_subset16_v1/"
    "shared_paired_endpoint_fit_dev_and_route_embeddings"
)
DEFAULT_RJ_BASE = Path(
    f"{DATA_ROOT}/papc/upstream_models/"
    "distilbert-base-uncased-12040acc"
)
DEFAULT_OUTPUT = Path(f"{DATA_ROOT}/papc/agserve_qac_v1/models")


@dataclass(frozen=True)
class Example:
    key: str
    text: str
    label: int


class TextDataset:
    def __init__(self, encodings: dict[str, Any], labels: Sequence[int]) -> None:
        self.encodings = encodings
        self.labels = tuple(int(label) for label in labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-dev16-root", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--state-rj-dataset", type=Path, default=DEFAULT_STATE_RJ)
    parser.add_argument("--agent-rj-dataset", type=Path, default=DEFAULT_AGENT_RJ)
    parser.add_argument(
        "--profile",
        choices=("state_agent", "tau2_retail"),
        default="state_agent",
    )
    parser.add_argument(
        "--retail-paired-root",
        type=Path,
        default=DEFAULT_RETAIL_PAIRED,
    )
    parser.add_argument("--qj-base-model", default="google-bert/bert-base-uncased")
    parser.add_argument("--rj-base-model", type=Path, default=DEFAULT_RJ_BASE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--qj-epochs", type=int, default=10)
    parser.add_argument("--rj-epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError("agserve_fit_output_exists_no_overwrite")
    if args.qj_epochs != 10 or args.rj_epochs != 6 or args.seed != 2024:
        raise ValueError("agserve_fit_upstream_training_schedule_drifted")
    if args.learning_rate != 2e-5:
        raise ValueError("agserve_fit_learning_rate_drifted")

    _seed_everything(args.seed)
    if args.profile == "tau2_retail":
        qj_train, qj_cal, qj_sources = _load_retail_qj_examples(
            args.retail_paired_root
        )
        rj_train, rj_cal, rj_sources = _load_retail_rj_examples(
            args.retail_paired_root
        )
    else:
        qj_train, qj_cal, qj_sources = _load_qj_examples(args.shared_dev16_root)
        rj_train, rj_cal, rj_sources = _load_rj_examples(
            args.state_rj_dataset,
            args.agent_rj_dataset,
        )
    output.mkdir(parents=True, exist_ok=False)
    qj_dir = output / "q_judge"
    rj_dir = output / "r_judge"
    qj_metrics, qj_probs = _fit_classifier(
        base_model=args.qj_base_model,
        output_dir=qj_dir,
        train_examples=qj_train,
        calibration_examples=qj_cal,
        max_length=AGSERVE_QJ_MAX_LENGTH,
        epochs=args.qj_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        oversample_positive=True,
        negative_loss_multiplier=1.25,
    )
    rj_metrics, rj_probs = _fit_classifier(
        base_model=str(Path(args.rj_base_model).resolve()),
        output_dir=rj_dir,
        train_examples=rj_train,
        calibration_examples=rj_cal,
        max_length=AGSERVE_RJ_MAX_LENGTH,
        epochs=args.rj_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        oversample_positive=True,
        negative_loss_multiplier=1.0,
    )
    qj_threshold, qj_threshold_metrics = _zero_false_negative_threshold(
        qj_probs,
        [item.label for item in qj_cal],
    )
    rj_threshold, rj_threshold_metrics = _zero_false_negative_threshold(
        rj_probs,
        [item.label for item in rj_cal],
    )
    file_hashes = {
        str(path.relative_to(output)): _file_sha256(path)
        for directory in (qj_dir, rj_dir)
        for path in sorted(directory.iterdir())
        if path.name
        in {
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        }
    }
    bundle_sha = _digest(file_hashes)
    metadata = {
        "schema": "papc.agserve_qac_artifact.v1",
        "method": "AgServe-QAC",
        "scope": "quality_aware_client_only",
        "upstream_revision": AGSERVE_UPSTREAM_REVISION,
        "model_pair": {
            "cheap": "gpt-5.4-mini",
            "strong": "gpt-5.5",
            "reasoning_effort": "high",
        },
        "thresholds": {
            "q_judge": qj_threshold,
            "r_judge": rj_threshold,
            "selection": "heldout_in_domain_zero_false_negative_then_min_strong",
        },
        "check_frequencies": (
            {"state_bench": 4, "agent_diff": 1}
            if args.profile == "state_agent"
            else {"state_bench": 4, "agent_diff": 1, "tau2_retail": 2}
        ),
        "training_profile": args.profile,
        "training": {
            "q_judge": {
                "architecture": "bert_sequence_classifier",
                "base_model": args.qj_base_model,
                "train_count": len(qj_train),
                "calibration_count": len(qj_cal),
                "epochs": args.qj_epochs,
                "batch_size": 16,
                "max_length": AGSERVE_QJ_MAX_LENGTH,
                "seed": args.seed,
                "negative_loss_multiplier": 1.25,
                "metrics": qj_metrics,
                "threshold_metrics": qj_threshold_metrics,
            },
            "r_judge": {
                "architecture": "distilbert_sequence_classifier",
                "base_model": str(Path(args.rj_base_model).resolve()),
                "train_count": len(rj_train),
                "calibration_count": len(rj_cal),
                "epochs": args.rj_epochs,
                "batch_size": 16,
                "max_length": AGSERVE_RJ_MAX_LENGTH,
                "seed": args.seed,
                "metrics": rj_metrics,
                "threshold_metrics": rj_threshold_metrics,
            },
        },
        "source_bindings": {
            "q_judge": qj_sources,
            "r_judge": rj_sources,
            "official_test_examples_used": False,
        },
        "file_sha256s": file_hashes,
        "bundle_sha256": bundle_sha,
        "provider_calls": 0,
        "secret_reads": 0,
    }
    _write_json_no_replace(output / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


def _load_qj_examples(
    shared_root: Path,
) -> tuple[tuple[Example, ...], tuple[Example, ...], dict[str, Any]]:
    root = Path(shared_root).resolve()
    pairs: list[Example] = []
    source_files: list[Path] = []
    for surface in ("state_bench", "agent_diff"):
        cheap_path = root / surface / "cheap" / "dev16_rows.jsonl"
        cheap_rows = _read_jsonl(cheap_path)
        source_files.append(cheap_path)
        strong_path = root / surface / "strong" / "dev16_rows.jsonl"
        if strong_path.is_file():
            strong_rows = _read_jsonl(strong_path)
            source_files.append(strong_path)
        else:
            row_paths = sorted((root / surface / "strong" / "rows").glob("row_*.json"))
            strong_rows = [json.loads(path.read_text(encoding="utf-8")) for path in row_paths]
            source_files.extend(row_paths)
        strong_by_example = {row["example_sha256"]: row for row in strong_rows}
        for cheap in cheap_rows:
            strong = strong_by_example.get(cheap["example_sha256"])
            if strong is None:
                continue
            label = int(
                float(cheap["official_score"]) == 0.0
                and float(strong["official_score"]) == 1.0
            )
            pairs.append(
                Example(
                    key=f"{surface}:{cheap['example_sha256']}",
                    text=str(cheap["original_request"]),
                    label=label,
                )
            )
    positives = sorted((item for item in pairs if item.label == 1), key=lambda x: x.key)
    negatives = sorted((item for item in pairs if item.label == 0), key=lambda x: x.key)
    if len(positives) < 2 or len(negatives) < 8:
        raise ValueError("agserve_qj_paired_dev_too_small")
    calibration = (positives[-1], *negatives[::4][:5])
    calibration_keys = {item.key for item in calibration}
    train = tuple(item for item in pairs if item.key not in calibration_keys)
    if set(item.key for item in train) & calibration_keys:
        raise RuntimeError("agserve_qj_fit_cal_overlap")
    return (
        train,
        tuple(calibration),
        {
            "paired_examples": len(pairs),
            "positive_strong_only": len(positives),
            "train_keys_sha256": _digest(sorted(item.key for item in train)),
            "calibration_keys_sha256": _digest(sorted(calibration_keys)),
            "source_file_sha256s": {
                str(path): _file_sha256(path) for path in source_files
            },
        },
    )


def _load_rj_examples(
    state_path: Path,
    agent_path: Path,
) -> tuple[tuple[Example, ...], tuple[Example, ...], dict[str, Any]]:
    train: list[Example] = []
    calibration: list[Example] = []
    sources = []
    for surface, path in (
        ("state_bench", Path(state_path).resolve()),
        ("agent_diff", Path(agent_path).resolve()),
    ):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError("agserve_rj_dataset_not_list")
        sources.append(path)
        for index, row in enumerate(rows):
            label = int(row["label"])
            if label not in {0, 1}:
                raise ValueError("agserve_rj_label_not_binary")
            example = Example(
                key=f"{surface}:{row['task_name']}:{row['step_num']}:{index}",
                text=(
                    f"[CLS] smaller [SEP] in-domain agent task [SEP] "
                    f"{row['text']} [SEP]"
                ),
                label=label,
            )
            if row["split"] == "train":
                train.append(example)
            elif row["split"] == "test":
                calibration.append(example)
            else:
                raise ValueError("agserve_rj_split_invalid")
    if not train or not calibration:
        raise ValueError("agserve_rj_split_empty")
    return (
        tuple(train),
        tuple(calibration),
        {
            "train_count": len(train),
            "calibration_count": len(calibration),
            "train_labels": dict(Counter(item.label for item in train)),
            "calibration_labels": dict(Counter(item.label for item in calibration)),
            "source_file_sha256s": {
                str(path): _file_sha256(path) for path in sources
            },
        },
    )


def _load_retail_qj_examples(
    paired_root: Path,
) -> tuple[tuple[Example, ...], tuple[Example, ...], dict[str, Any]]:
    rows, source_files = _load_retail_paired_rows(paired_root)

    def examples(split: str) -> tuple[Example, ...]:
        cheap = rows[(split, "cheap")]
        strong = rows[(split, "strong")]
        if set(cheap) != set(strong):
            raise ValueError("agserve_retail_qj_pair_keys_mismatch")
        return tuple(
            Example(
                key=f"tau2_retail:{split}:{key}",
                text=str(cheap[key]["output"]["original_request"]),
                label=int(
                    float(cheap[key]["output"]["official_score"]) == 0.0
                    and float(strong[key]["output"]["official_score"]) == 1.0
                ),
            )
            for key in sorted(cheap)
        )

    train = examples("router_fit")
    calibration = examples("router_dev")
    if not any(item.label == 1 for item in train):
        raise ValueError("agserve_retail_qj_train_has_no_positive")
    if not any(item.label == 1 for item in calibration):
        raise ValueError("agserve_retail_qj_calibration_has_no_positive")
    return (
        train,
        calibration,
        {
            "paired_examples": len(train) + len(calibration),
            "fit_paired_examples": len(train),
            "dev_paired_examples": len(calibration),
            "positive_strong_only": sum(item.label for item in (*train, *calibration)),
            "train_keys_sha256": _digest(sorted(item.key for item in train)),
            "calibration_keys_sha256": _digest(
                sorted(item.key for item in calibration)
            ),
            "source_file_sha256s": {
                str(path): _file_sha256(path) for path in source_files
            },
        },
    )


def _load_retail_rj_examples(
    paired_root: Path,
) -> tuple[tuple[Example, ...], tuple[Example, ...], dict[str, Any]]:
    rows, source_files = _load_retail_paired_rows(paired_root)

    def examples(split: str) -> tuple[Example, ...]:
        built: list[Example] = []
        for example_sha, row in sorted(rows[(split, "cheap")].items()):
            output = row["output"]
            request = str(output["original_request"])
            label = int(float(output["official_score"]) == 0.0)
            history = output.get("public_episode")
            if not isinstance(history, list):
                raise TypeError("agserve_retail_public_episode_not_list")
            for turn_index, message in enumerate(history):
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                following = history[turn_index + 1] if turn_index + 1 < len(history) else {}
                response = json.dumps(
                    message,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                observation = json.dumps(
                    following,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                built.append(
                    Example(
                        key=f"tau2_retail:{split}:{example_sha}:{turn_index}",
                        text=(
                            f"[CLS] smaller [SEP] {request} [SEP] "
                            f"{response} [SEP] {observation} [SEP]"
                        ),
                        label=label,
                    )
                )
        return tuple(built)

    train = examples("router_fit")
    calibration = examples("router_dev")
    if not train or not calibration:
        raise ValueError("agserve_retail_rj_split_empty")
    if not any(item.label == 1 for item in calibration):
        raise ValueError("agserve_retail_rj_calibration_has_no_positive")
    return (
        train,
        calibration,
        {
            "train_count": len(train),
            "calibration_count": len(calibration),
            "train_labels": dict(Counter(item.label for item in train)),
            "calibration_labels": dict(Counter(item.label for item in calibration)),
            "label_semantics": "cheap_episode_failure_projected_to_public_response_turns",
            "source_file_sha256s": {
                str(path): _file_sha256(path) for path in source_files
            },
        },
    )


def _load_retail_paired_rows(
    paired_root: Path,
) -> tuple[dict[tuple[str, str], dict[str, dict[str, Any]]], tuple[Path, ...]]:
    root = Path(paired_root).resolve()
    loaded: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    source_files: list[Path] = []
    for split in ("router_fit", "router_dev"):
        for role in ("cheap", "strong"):
            matches = tuple(
                root.glob(
                    f"*shared_{role}_endpoint.tau2_retail.{split}.jsonl"
                )
            )
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"agserve_retail_paired_source_missing:{split}:{role}"
                )
            path = matches[0]
            source_files.append(path)
            items = {
                row["example_sha256"]: row
                for row in _read_jsonl(path)
                if row.get("record_type") == "item"
            }
            expected = 59 if split == "router_fit" else 15
            if len(items) != expected:
                raise ValueError("agserve_retail_paired_denominator_mismatch")
            loaded[(split, role)] = items
    return loaded, tuple(source_files)


def _fit_classifier(
    *,
    base_model: str,
    output_dir: Path,
    train_examples: Sequence[Example],
    calibration_examples: Sequence[Example],
    max_length: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
    oversample_positive: bool,
    negative_loss_multiplier: float,
) -> tuple[dict[str, Any], list[float]]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
    ).to(torch.device(device))
    expanded = list(train_examples)
    if oversample_positive:
        negatives = sum(item.label == 0 for item in expanded)
        positives = [item for item in expanded if item.label == 1]
        if positives:
            repeat = max(1, negatives // len(positives))
            expanded.extend(positives * (repeat - 1))
    random.Random(seed).shuffle(expanded)
    encoded = tokenizer(
        [item.text for item in expanded],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = TextDataset(encoded, [item.label for item in expanded])
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    class_weights = torch.tensor(
        [negative_loss_multiplier, 1.0],
        dtype=torch.float32,
        device=torch.device(device),
    )
    model.train()
    losses: list[float] = []
    for _ in range(epochs):
        for batch in loader:
            labels = batch.pop("labels").to(torch.device(device))
            inputs = {key: value.to(torch.device(device)) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
                weight=class_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    probabilities = _predict_probabilities(
        model=model,
        tokenizer=tokenizer,
        examples=calibration_examples,
        max_length=max_length,
        device=device,
    )
    predictions = [int(value >= 0.5) for value in probabilities]
    labels = [item.label for item in calibration_examples]
    return (
        {
            "final_train_loss": round(losses[-1], 8),
            "calibration_accuracy_at_0_5": round(
                sum(a == b for a, b in zip(predictions, labels, strict=True))
                / len(labels),
                6,
            ),
            "calibration_positive_probability_min": round(min(probabilities), 8),
            "calibration_positive_probability_max": round(max(probabilities), 8),
        },
        probabilities,
    )


def _predict_probabilities(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[Example],
    max_length: int,
    device: str,
) -> list[float]:
    import torch

    model.eval()
    encoded = tokenizer(
        [item.text for item in examples],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {key: value.to(torch.device(device)) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        values = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
    return [float(value) for value in values]


def _zero_false_negative_threshold(
    probabilities: Sequence[float],
    labels: Sequence[int],
) -> tuple[float, dict[str, Any]]:
    positives = [p for p, label in zip(probabilities, labels, strict=True) if label == 1]
    if not positives:
        raise ValueError("agserve_calibration_has_no_positive")
    threshold = max(0.0, min(1.0, min(positives)))
    predictions = [int(value >= threshold) for value in probabilities]
    false_negative = sum(
        prediction == 0 and label == 1
        for prediction, label in zip(predictions, labels, strict=True)
    )
    false_positive = sum(
        prediction == 1 and label == 0
        for prediction, label in zip(predictions, labels, strict=True)
    )
    return (
        threshold,
        {
            "false_negatives": false_negative,
            "false_positives": false_positive,
            "selected_strong_or_bad": sum(predictions),
            "count": len(labels),
        },
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
    except ImportError:
        pass


def _write_json_no_replace(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(target, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
