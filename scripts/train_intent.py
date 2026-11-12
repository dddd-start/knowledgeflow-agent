"""微调 TinyBERT 做意图分类(6 类)—— 用自建数据集,CPU 约 2~4 分钟。

为什么用 TinyBERT 而不是直接问大模型?
    "现在几点""你好"这类请求占实际流量的一大块,让大模型判意图要 1~2 秒且要花钱;
    4 层中文 TinyBERT 只有 ~50MB、CPU 推理 <10ms,效果对 6 类封闭意图足够。
    这就是意图路由能把简单请求做到"0 次 LLM 调用"的前提。

训练策略:
- 冻结?不冻结。4 层小模型全参微调,660 条样本 8 个 epoch 足够收敛,CPU 也快;
- 按 eval.json 选最佳 epoch(不 shuffle 的固定验证集,结果可复现);
- 产物:model.safetensors + tokenizer + labels.json(推理端直接读)。

用法:
    python scripts/train_intent.py --epochs 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import BertForSequenceClassification, BertTokenizer  # noqa: E402

from app.config import settings  # noqa: E402
from app.obs.logging import setup_logging  # noqa: E402

BASE_MODEL = "huawei-noah/TinyBERT_4L_zh"
MAX_LEN = 32


class IntentDataset(Dataset):
    def __init__(self, items: list[dict], label2id: dict[str, int], tokenizer) -> None:
        self.items = items
        self.label2id = label2id
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        enc = self.tokenizer(
            item["text"], truncation=True, max_length=MAX_LEN, padding="max_length"
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "token_type_ids": torch.tensor(enc.get("token_type_ids", [0] * MAX_LEN)),
            "labels": torch.tensor(self.label2id[item["intent"]]),
        }


@torch.no_grad()
def evaluate(model, loader) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).logits
        pred = logits.argmax(dim=-1)
        correct += int((pred == batch["labels"]).sum())
        total += len(batch["labels"])
    return correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    setup_logging("INFO")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = PROJECT_ROOT / "data" / "intents"
    train_items = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))
    eval_items = json.loads((data_dir / "eval.json").read_text(encoding="utf-8"))
    labels = json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))
    label_ids = [item["id"] for item in labels]
    label2id = {label: i for i, label in enumerate(label_ids)}
    unknown = {item["intent"] for item in train_items + eval_items} - set(label_ids)
    if unknown:
        print(f"[错误] 数据集里出现未定义意图: {unknown}")
        return 1

    print(f"train={len(train_items)} eval={len(eval_items)} labels={label_ids}")
    # TinyBERT_4L_zh 的 tokenizer_config 指向 fast 格式,transformers 5.x 直接加载会失败,
    # 用 slow 版 BertTokenizer(vocab.txt),训练/推理都稳定。
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    # 上游 config.json 没有 model_type,AutoModel 推断失败;直接指定 BERT 架构
    model = BertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(label_ids))

    train_loader = DataLoader(
        IntentDataset(train_items, label2id, tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(IntentDataset(eval_items, label2id, tokenizer), batch_size=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
    )

    out_dir = Path(settings.intent_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=batch["labels"],
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(out.loss)
        acc = evaluate(model, eval_loader)
        print(
            f"epoch {epoch:>2}/{args.epochs} loss={total_loss / len(train_loader):.4f} "
            f"eval_acc={acc:.4f}{'  <- best' if acc > best_acc else ''}"
        )
        if acc >= best_acc:
            best_acc = acc
            model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            (out_dir / "labels.json").write_text(
                json.dumps(
                    {"labels": label_ids, "base_model": BASE_MODEL, "eval_acc": round(acc, 4)},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    elapsed = time.perf_counter() - started
    print(f"\n训练完成: {elapsed:.0f}s, best eval_acc={best_acc:.4f}")
    print(f"模型已保存到 {out_dir}(推理端 INTENT_BACKEND=auto 会自动加载)")
    metrics = {
        "base_model": BASE_MODEL,
        "train_size": len(train_items),
        "eval_size": len(eval_items),
        "epochs": args.epochs,
        "best_eval_acc": round(best_acc, 4),
        "train_seconds": round(elapsed, 1),
    }
    (PROJECT_ROOT / "eval").mkdir(exist_ok=True)
    (PROJECT_ROOT / "eval" / "train-intent.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
