#!/usr/bin/env python3
"""Compare C++ CPU logits against Hugging Face Transformers for one prompt.

The script reads token ids from the C++ debug dump and feeds those exact ids
to Transformers. This keeps tokenizer differences out of the numerical check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./build/manual/toyllm")
    parser.add_argument("--model", default="models/qwen3-0.6b")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--dump-dir", default="build/cpu-transformers-compare")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--max-abs-tol", type=float, default=None)
    parser.add_argument("--mean-abs-tol", type=float, default=None)
    return parser.parse_args()


def import_deps():
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        print(f"missing Python dependency: {error}", file=sys.stderr)
        print("install torch, transformers, and numpy to run this alignment script", file=sys.stderr)
        raise SystemExit(2) from error
    return np, torch, AutoModelForCausalLM


def run_cpp(args: argparse.Namespace) -> None:
    command = [
        args.binary,
        "infer",
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dump-dir",
        args.dump_dir,
    ]
    subprocess.run(command, check=True)


def load_records(dump_dir: Path) -> dict[str, dict]:
    metadata_path = dump_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise SystemExit(f"missing dump metadata: {metadata_path}")

    records: dict[str, dict] = {}
    with metadata_path.open("r", encoding="utf-8") as metadata:
        for line in metadata:
            record = json.loads(line)
            records[record["name"]] = record
    return records


def load_array(np, dump_dir: Path, record: dict):
    dtype = {"f32": np.float32, "i64": np.int64}[record["dtype"]]
    values = np.fromfile(dump_dir / record["file"], dtype=dtype)
    return values.reshape(record["shape"])


def main() -> int:
    args = parse_args()
    np, torch, AutoModelForCausalLM = import_deps()

    dump_dir = Path(args.dump_dir)
    if not args.skip_run:
        run_cpp(args)

    records = load_records(dump_dir)
    prompt_tokens = load_array(np, dump_dir, records["prompt_tokens"]).astype(np.int64)
    logits_name = f"position.{len(prompt_tokens) - 1}.logits"
    cpp_logits = load_array(np, dump_dir, records[logits_name]).astype(np.float32)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    input_ids = torch.tensor([prompt_tokens.tolist()], dtype=torch.long)
    with torch.no_grad():
        hf_logits = model(input_ids=input_ids).logits[0, -1].detach().cpu().float().numpy()

    diff = np.abs(cpp_logits - hf_logits)
    cpp_top = int(np.argmax(cpp_logits))
    hf_top = int(np.argmax(hf_logits))

    print(f"prompt_tokens: {len(prompt_tokens)}")
    print(f"cpp_top_token: {cpp_top}")
    print(f"hf_top_token: {hf_top}")
    print(f"top_match: {cpp_top == hf_top}")
    print(f"max_abs_diff: {float(diff.max()):.8f}")
    print(f"mean_abs_diff: {float(diff.mean()):.8f}")

    if args.max_abs_tol is not None and float(diff.max()) > args.max_abs_tol:
        print("max_abs_diff exceeds tolerance", file=sys.stderr)
        return 1
    if args.mean_abs_tol is not None and float(diff.mean()) > args.mean_abs_tol:
        print("mean_abs_diff exceeds tolerance", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
