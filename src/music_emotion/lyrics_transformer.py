from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .results_log import Experiment, log_experiment

PROCESSED = Path("data/processed")
RAW = Path("data/raw")
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
SEED = 42


def read_split(root: Path, name: str) -> set[str]:
    path = root / "tvt_dataframes" / "tvt_70_15_15" / f"tvt_70_15_15_{name}_lyrics_complete.csv"
    return set(pd.read_csv(path)["Song"])


def load_lyrics_split(raw_dir: Path, processed_dir: Path):
    frame = pd.read_csv(processed_dir / "merge_lyrics.csv")
    root = raw_dir / "merge" / "MERGE_Lyrics_Complete"
    train_ids = read_split(root, "train") | read_split(root, "validate")
    test_ids = read_split(root, "test")
    frame["text"] = [
        (raw_dir / path).read_text(encoding="utf-8", errors="replace")
        for path in frame["lyrics_path"]
    ]
    train = frame[frame["sample_id"].isin(train_ids)].reset_index(drop=True)
    test = frame[frame["sample_id"].isin(test_ids)].reset_index(drop=True)
    return train, test


def metric_block(actual, predicted) -> dict:
    labels = sorted(set(actual))
    per_class = f1_score(actual, predicted, labels=labels, average=None, zero_division=0)
    return {
        "accuracy_mean": round(accuracy_score(actual, predicted), 4),
        "macro_f1_mean": round(f1_score(actual, predicted, average="macro", zero_division=0), 4),
        "balanced_acc_mean": round(balanced_accuracy_score(actual, predicted), 4),
        "per_class_f1": {label: round(float(s), 4) for label, s in zip(labels, per_class)},
    }


def linear_reference(train, test) -> dict:
    model = Pipeline(
        [
            (
                "tfidf",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb", ngram_range=(3, 5), max_features=20000
                            ),
                        ),
                    ]
                ),
            ),
            ("model", LinearSVC(C=1.0)),
        ]
    )
    model.fit(train["text"], train["quadrant"])
    predicted = model.predict(test["text"])
    return metric_block(test["quadrant"].tolist(), list(predicted))


def train_transformer(train, test, epochs: int, batch_size: int) -> dict:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(SEED)
    labels = sorted(train["quadrant"].unique())
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode(texts):
        return tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LENGTH)

    class LyricsDataset(torch.utils.data.Dataset):
        def __init__(self, texts, label_ids):
            self.encodings = encode(texts)
            self.label_ids = label_ids

        def __len__(self):
            return len(self.label_ids)

        def __getitem__(self, index):
            item = {k: torch.tensor(v[index]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.label_ids[index])
            return item

    train_ds = LyricsDataset(train["text"], [label_to_id[q] for q in train["quadrant"]])
    test_ds = LyricsDataset(test["text"], [label_to_id[q] for q in test["quadrant"]])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))

    args = TrainingArguments(
        output_dir="results/_distilbert",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()

    logits = trainer.predict(test_ds).predictions
    predicted_ids = np.argmax(logits, axis=1)
    predicted = [id_to_label[i] for i in predicted_ids]
    actual = test["quadrant"].tolist()
    block = metric_block(actual, predicted)
    block["_report"] = classification_report(actual, predicted, zero_division=0)
    return block


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train, test = load_lyrics_split(args.raw_dir, args.processed_dir)
    print(f"train lyrics: {len(train)} | test lyrics: {len(test)}")

    reference = linear_reference(train, test)
    print(
        f"[reference] LinearSVC (same split): acc={reference['accuracy_mean']:.3f} "
        f"macroF1={reference['macro_f1_mean']:.3f}"
    )
    log_experiment(
        Experiment(
            modality="transformer",
            name="LinearSVC reference (official split)",
            config="MERGE official 70/15/15; TF-IDF word+char -> LinearSVC",
            metrics=reference,
        )
    )

    print("Fine-tuning DistilBERT (this is the slow part on CPU)...")
    transformer = train_transformer(train, test, args.epochs, args.batch_size)
    report = transformer.pop("_report")
    print(
        f"[transformer] DistilBERT: acc={transformer['accuracy_mean']:.3f} "
        f"macroF1={transformer['macro_f1_mean']:.3f}"
    )
    print(report)
    log_experiment(
        Experiment(
            modality="transformer",
            name=f"DistilBERT fine-tuned ({args.epochs} epochs)",
            config=f"MERGE official 70/15/15; {MODEL_NAME}, max_len={MAX_LENGTH}",
            metrics=transformer,
        )
    )


if __name__ == "__main__":
    main()
