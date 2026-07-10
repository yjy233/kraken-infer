#!/usr/bin/env python3
"""Run an end-to-end Qwen3.5 0.8B MTP request through the OpenAI gateway."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def wait_for_health(url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - surface the final failure below.
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"gateway did not become healthy: {last_error}")


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            response_body = response.read().decode("utf-8")
            return json.loads(response_body), headers
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"gateway returned HTTP {exc.code}: {error_body}") from exc


def int_header(headers: dict[str, str], name: str) -> int:
    key = name.lower()
    if key not in headers:
        raise RuntimeError(f"missing response header: {name}")
    try:
        return int(headers[key])
    except ValueError as exc:
        raise RuntimeError(f"header {name} is not an integer: {headers[key]}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./build/debug/kraken-infer")
    parser.add_argument("--model", default="models/qwen3.5-0.8b-mtp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18184)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--p-min", type=float, default=0.0)
    parser.add_argument(
        "--prompt",
        default="Write one short sentence about Metal acceleration.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    binary = Path(args.binary)
    model = Path(args.model)
    for label, path in {"binary": binary, "model": model}.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    server = subprocess.Popen(
        [
            str(binary),
            "serve",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--model",
            str(model),
            "--model-id",
            "qwen35-mtp-test",
            "--device",
            "mps",
            "--max-new-tokens",
            str(args.max_tokens),
            "--mtp",
            "--mtp-draft-tokens",
            str(args.draft_tokens),
            "--mtp-p-min",
            str(args.p_min),
            "--no-cache-prompt",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base_url = f"http://{args.host}:{args.port}"
        wait_for_health(f"{base_url}/health", 15.0)

        payload = {
            "model": "qwen35-mtp-test",
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "mtp": True,
            "mtp_draft_tokens": args.draft_tokens,
            "mtp_p_min": args.p_min,
        }
        response, headers = post_json(
            f"{base_url}/v1/completions",
            payload,
            args.timeout,
        )
        content = response["choices"][0].get("text", "").strip()
        if not content:
            raise RuntimeError(f"empty completion text: {json.dumps(response)[:1000]}")

        enabled = int_header(headers, "X-Kraken-MTP-Enabled")
        layers = int_header(headers, "X-Kraken-MTP-Layers")
        drafted = int_header(headers, "X-Kraken-MTP-Drafted-Tokens")
        verify_steps = int_header(headers, "X-Kraken-MTP-Verify-Steps")
        confidence_stops = int_header(headers, "X-Kraken-MTP-Confidence-Stops")
        if enabled != 1:
            raise RuntimeError(f"MTP was not enabled; headers={headers}")
        if layers < 1:
            raise RuntimeError(f"MTP layer count was not reported; headers={headers}")
        if drafted < 1 and confidence_stops < 1:
            raise RuntimeError(f"MTP did not draft or confidence-stop any tokens; headers={headers}")
        if drafted > 0 and verify_steps < 1:
            raise RuntimeError(f"MTP verification did not run; headers={headers}")

        print(content)
        print(
            json.dumps(
                {
                    "mtp_enabled": enabled,
                    "mtp_layers": layers,
                    "draft_tokens": int_header(headers, "X-Kraken-MTP-Draft-Tokens"),
                    "p_min": headers.get("x-kraken-mtp-p-min"),
                    "drafted_tokens": drafted,
                    "accepted_tokens": int_header(headers, "X-Kraken-MTP-Accepted-Tokens"),
                    "verify_steps": verify_steps,
                    "confidence_stops": confidence_stops,
                    "adaptive_budget": int_header(
                        headers, "X-Kraken-MTP-Adaptive-Budget"
                    ),
                    "adaptive_changes": int_header(
                        headers, "X-Kraken-MTP-Adaptive-Changes"
                    ),
                    "verified_by_position": headers.get(
                        "x-kraken-mtp-verified-by-position", ""
                    ),
                    "accepted_by_position": headers.get(
                        "x-kraken-mtp-accepted-by-position", ""
                    ),
                    "usage": response.get("usage", {}),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    finally:
        server.terminate()
        try:
            stdout, _ = server.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            stdout, _ = server.communicate(timeout=5)
        if server.returncode not in {0, -15, -2, 1} and stdout:
            print(stdout, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
