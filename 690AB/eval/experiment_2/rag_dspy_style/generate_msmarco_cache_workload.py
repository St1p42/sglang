# generate_msmarco_cache_workload_slim.py
#
# Generates MS MARCO-derived shared-prefix JSONL workloads for SGLang
# cache-aware scheduling benchmarks.
#
# Output schema per JSONL row:
#   {
#     "request_id": "...",
#     "group_id": "...",
#     "variant_id": 0,
#     "prompt": "Context:\n...\n\nQuestion: ...\nAnswer:",
#     "answer": "..."
#   }
#
# Colab:
#   !pip install datasets -q
#   !python generate_msmarco_cache_workload_slim.py --num-groups 64 --group-size 8 --out-dir /content
#
# Local:
#   pip install datasets
#   python generate_msmarco_cache_workload_slim.py --num-groups 64 --group-size 8 --out-dir ./workloads

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


PROMPT_TEMPLATE = """Context:
{context}

Question: {question}
Answer:"""


QUESTION_SUFFIXES = [
    "",
    " Answer concisely.",
    " Use only the provided context.",
    " Give a direct answer.",
    " Respond in one short sentence.",
    " Avoid extra explanation.",
    " Provide the answer only.",
    " Be brief.",
    " Answer based on the context.",
    " Keep the answer short.",
    " Do not add extra details.",
    " Use the passage above.",
    " Answer directly.",
    " Give only the final answer.",
    " Use evidence from the context.",
    " Be precise.",
]


def normalize_answer(answers: Any) -> str:
    """Return a single ground-truth answer string for optional eval."""
    if answers is None:
        return ""

    if isinstance(answers, list):
        for answer in answers:
            if answer is not None and str(answer).strip():
                return str(answer).strip()
        return ""

    return str(answers).strip()


def extract_passages(row: Dict[str, Any], selected_first: bool = True) -> List[str]:
    """
    Extract MS MARCO passage texts.

    Expected HF v1.1 shape:
      row["passages"] = {
        "is_selected": [...],
        "passage_text": [...],
        "url": [...]
      }
    """
    passages = row.get("passages")
    if not isinstance(passages, dict):
        return []

    texts = passages.get("passage_text") or []
    selected = passages.get("is_selected") or []

    if not isinstance(texts, list):
        texts = [texts]

    if not isinstance(selected, list):
        selected = [selected] * len(texts)

    if not selected_first:
        return [str(text).strip() for text in texts if text and str(text).strip()]

    selected_passages: List[str] = []
    unselected_passages: List[str] = []

    for i, text in enumerate(texts):
        if not text or not str(text).strip():
            continue

        text = str(text).strip()
        is_selected = False

        if i < len(selected):
            try:
                is_selected = int(selected[i]) == 1
            except Exception:
                is_selected = False

        if is_selected:
            selected_passages.append(text)
        else:
            unselected_passages.append(text)

    # Relevant/selected passages first, then fill with the rest.
    return selected_passages + unselected_passages


def build_context(
    passages: List[str],
    target_context_chars: int,
    max_context_chars: int,
) -> str:
    """
    Build one long context from MS MARCO passages.

    Uses chars rather than tokens so workload generation does not need a model tokenizer.
    """
    chunks: List[str] = []
    total_chars = 0

    for passage in passages:
        passage = passage.strip()
        if not passage:
            continue

        chunks.append(passage)
        total_chars += len(passage)

        if total_chars >= target_context_chars:
            break

    context = "\n\n".join(chunks).strip()

    if len(context) > max_context_chars:
        context = context[:max_context_chars].rstrip()

    return context


def make_question_variant(question: str, variant_idx: int) -> str:
    """
    Keep the real MS MARCO question mostly unchanged.
    Add small suffixes at the end so the long prompt prefix remains shared.
    """
    return question.strip() + QUESTION_SUFFIXES[variant_idx % len(QUESTION_SUFFIXES)]


def make_slim_item(
    group_id: str,
    variant_id: int,
    context: str,
    question: str,
    answer: str,
) -> Dict[str, Any]:
    """
    This is the agreed benchmark-facing schema.
    """
    return {
        "request_id": f"{group_id}_q{variant_id:02d}",
        "group_id": group_id,
        "variant_id": variant_id,
        "prompt": PROMPT_TEMPLATE.format(context=context, question=question),
        "answer": answer,
    }


def grouped_order(groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    # A1 A2 A3 ... B1 B2 B3 ...
    return [row for group in groups for row in group]


def interleaved_order(groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    # A1 B1 C1 ... A2 B2 C2 ...
    if not groups:
        return []

    group_size = len(groups[0])
    rows: List[Dict[str, Any]] = []

    for variant_idx in range(group_size):
        for group in groups:
            rows.append(group[variant_idx])

    return rows


def shuffled_order(groups: List[List[Dict[str, Any]]], seed: int) -> List[Dict[str, Any]]:
    # Deterministic random order.
    rows = [row for group in groups for row in group]
    random.Random(seed).shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows):,} rows: {path}")


def make_groups(args: argparse.Namespace) -> List[List[Dict[str, Any]]]:
    print(f"Loading dataset: {args.dataset_name}, config={args.config}, split={args.split}")
    print("Using streaming=True so this works well in Colab.")

    ds = load_dataset(
        args.dataset_name,
        args.config,
        split=args.split,
        streaming=True,
        trust_remote_code=False,
    )

    groups: List[List[Dict[str, Any]]] = []
    seen_contexts = set()
    scanned = 0
    skipped = 0

    for row in ds:
        scanned += 1

        query = str(row.get("query", "")).strip()
        answer = normalize_answer(row.get("answers"))

        if not query or not answer:
            skipped += 1
            continue

        passages = extract_passages(row, selected_first=True)
        if not passages:
            skipped += 1
            continue

        context = build_context(
            passages=passages,
            target_context_chars=args.target_context_chars,
            max_context_chars=args.max_context_chars,
        )

        if len(context) < args.min_context_chars:
            skipped += 1
            continue

        # Avoid accidentally creating duplicate groups with identical contexts.
        context_key = context[:500]
        if context_key in seen_contexts:
            skipped += 1
            continue
        seen_contexts.add(context_key)

        group_id = f"doc_{len(groups):05d}"
        group: List[Dict[str, Any]] = []

        for variant_id in range(args.group_size):
            question = make_question_variant(query, variant_id)
            group.append(
                make_slim_item(
                    group_id=group_id,
                    variant_id=variant_id,
                    context=context,
                    question=question,
                    answer=answer,
                )
            )

        groups.append(group)

        if len(groups) % 10 == 0:
            print(
                f"Built {len(groups):,}/{args.num_groups:,} groups... "
                f"scanned={scanned:,}, skipped={skipped:,}"
            )

        if len(groups) >= args.num_groups:
            break

        if args.max_scan_rows is not None and scanned >= args.max_scan_rows:
            break

    print()
    print("Generation summary")
    print("------------------")
    print(f"Groups built:       {len(groups):,}")
    print(f"Group size:         {args.group_size:,}")
    print(f"Total requests:     {len(groups) * args.group_size:,}")
    print(f"Rows scanned:       {scanned:,}")
    print(f"Rows skipped:       {skipped:,}")

    if len(groups) < args.num_groups:
        raise RuntimeError(
            f"Only built {len(groups)} groups, but requested {args.num_groups}. "
            f"Try lowering --min-context-chars or increasing --max-scan-rows."
        )

    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate slim MS MARCO-derived shared-prefix JSONL workloads "
            "for SGLang cache-aware scheduling benchmarks."
        )
    )

    parser.add_argument("--dataset-name", type=str, default="microsoft/ms_marco")
    parser.add_argument("--config", type=str, default="v1.1")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--num-groups", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)

    parser.add_argument("--min-context-chars", type=int, default=1200)
    parser.add_argument("--target-context-chars", type=int, default=3000)
    parser.add_argument("--max-context-chars", type=int, default=4500)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--prefix", type=str, default="workload_msmarco")

    parser.add_argument(
        "--max-scan-rows",
        type=int,
        default=None,
        help="Optional cap on scanned dataset rows. Usually leave unset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    groups = make_groups(args)
    suffix = f"{args.num_groups}x{args.group_size}"

    write_jsonl(
        out_dir / f"{args.prefix}_grouped_{suffix}.jsonl",
        grouped_order(groups),
    )

    write_jsonl(
        out_dir / f"{args.prefix}_interleaved_{suffix}.jsonl",
        interleaved_order(groups),
    )

    write_jsonl(
        out_dir / f"{args.prefix}_shuffled_{suffix}_seed{args.seed}.jsonl",
        shuffled_order(groups, seed=args.seed),
    )

    print()
    print("Recommended first benchmark file:")
    print(out_dir / f"{args.prefix}_interleaved_{suffix}.jsonl")


if __name__ == "__main__":
    main()
