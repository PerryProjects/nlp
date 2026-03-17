"""Export selected Hugging Face models to local ONNX directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

MODEL_EXPORTS: dict[str, str] = {
    "classifier": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "sentiment": "nlptown/bert-base-multilingual-uncased-sentiment",
    "toxicity": "unitary/multilingual-toxic-xlm-roberta",
}


def export_model(model_id: str, output_dir: Path) -> None:
    """Download and export one sequence classification model to ONNX."""
    from optimum.onnxruntime import ORTModelForSequenceClassification

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)

    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(output_dir)

    print(f"exported {model_id} -> {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export ONNX models for NLP microservice")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/models"),
        help="Directory where ONNX model folders are created",
    )
    return parser.parse_args()


def main() -> None:
    """Export all configured models."""
    args = parse_args()

    for name, model_id in MODEL_EXPORTS.items():
        export_model(model_id=model_id, output_dir=args.output_dir / name)


if __name__ == "__main__":
    main()
