"""意图分类评测 —— 对比「规则分类器」与「微调 TinyBERT」,输出准确率/各类别 P-R-F1/混淆矩阵。

用法:
    python scripts/eval_intent.py
输出:eval/report-intent.json + 终端表格
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.nlu.intent import RuleClassifier, TinyBertClassifier  # noqa: E402
from app.obs.logging import setup_logging  # noqa: E402


def per_class_report(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    report: dict = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred, strict=False) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred, strict=False) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred, strict=False) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(1 for g in gold if g == label),
        }
    return report


def confusion_matrix(gold: list[str], pred: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred, strict=False):
        matrix[index[g]][index[p]] += 1
    return matrix


def evaluate(name: str, classifier, items: list[dict]) -> dict:
    started = time.perf_counter()
    gold = [item["intent"] for item in items]
    pred = [classifier.predict(item["text"])[0] for item in items]
    elapsed = time.perf_counter() - started
    labels = sorted(set(gold) | set(pred))
    correct = sum(1 for g, p in zip(gold, pred, strict=False) if g == p)
    return {
        "classifier": name,
        "accuracy": round(correct / len(items), 4),
        "macro_f1": round(
            sum(v["f1"] for v in per_class_report(gold, pred, labels).values()) / len(labels), 4
        ),
        "per_class": per_class_report(gold, pred, labels),
        "confusion": confusion_matrix(gold, pred, labels),
        "labels": labels,
        "latency_ms_per_sample": round(elapsed * 1000 / len(items), 3),
        "misses": [
            {"text": item["text"], "gold": g, "pred": p}
            for item, g, p in zip(items, gold, pred, strict=False)
            if g != p
        ][:10],
    }


def main() -> int:
    setup_logging("WARNING")
    data_dir = PROJECT_ROOT / "data" / "intents"
    items = json.loads((data_dir / "eval.json").read_text(encoding="utf-8"))
    model_dir = Path(settings.intent_model_dir)

    results = []
    print(f"评测集: {len(items)} 条\n")
    results.append(evaluate("rule", RuleClassifier(), items))
    print(
        f"{'rule':<10} accuracy={results[-1]['accuracy']:.4f} "
        f"macro_f1={results[-1]['macro_f1']:.4f} "
        f"{results[-1]['latency_ms_per_sample']}ms/sample"
    )

    if TinyBertClassifier.is_available(model_dir):
        tinybert = TinyBertClassifier(model_dir)
        results.append(evaluate("tinybert", tinybert, items))
        print(
            f"{'tinybert':<10} accuracy={results[-1]['accuracy']:.4f} "
            f"macro_f1={results[-1]['macro_f1']:.4f} "
            f"{results[-1]['latency_ms_per_sample']}ms/sample"
        )
    else:
        print(f"(未找到微调模型 {model_dir},先运行 python scripts/train_intent.py)")

    report = {"eval_size": len(items), "results": results}
    out = PROJECT_ROOT / "eval"
    out.mkdir(exist_ok=True)
    (out / "report-intent.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n报告已写入 {out / 'report-intent.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
