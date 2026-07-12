#!/usr/bin/env python3
"""Run an end-to-end Qwen3.5 0.8B vision request through the OpenAI gateway."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z8BQDwAFgwJ/"
    "lV9SogAAAABJRU5ErkJggg=="
)


def lower_headers(headers) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def image_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


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


def post_json(url: str, payload: dict, timeout_seconds: float) -> tuple[dict, dict[str, str]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            headers = lower_headers(response.headers)
            return json.loads(response.read().decode("utf-8")), headers
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"gateway returned HTTP {exc.code}: {error_body}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./build/debug/kraken-infer")
    parser.add_argument("--model", default="models/qwen3.5-0.8b")
    parser.add_argument(
        "--mmproj",
        default="models/qwen3.5-0.8b/mmproj-Qwen3.5-0.8B-BF16.gguf",
    )
    parser.add_argument(
        "--image",
        default="",
        help="optional image path; omitted uses an embedded 1x1 PNG smoke image",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18183)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt",
        default="Describe the image in one sentence.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="optional sampling temperature; omit for greedy MTP-compatible decode",
    )
    parser.add_argument(
        "--expect-mtp-disabled-reason",
        default="",
        help="assert that MTP is available but disabled with this reason",
    )
    parser.add_argument(
        "--expect-mtp-enabled",
        action="store_true",
        help="assert that the gateway enables native MTP for the VL request",
    )
    parser.add_argument("--mtp-draft-tokens", type=int, default=3)
    parser.add_argument("--mtp-p-min", type=float, default=0.30)
    args = parser.parse_args()

    binary = Path(args.binary)
    model = Path(args.model)
    mmproj = Path(args.mmproj)
    image = Path(args.image) if args.image else None
    for label, path in {
        "binary": binary,
        "model": model,
        "mmproj": mmproj,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if image is not None and not image.exists():
        raise FileNotFoundError(f"image not found: {image}")

    command = [
        str(binary),
        "serve",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        str(model),
        "--model-id",
        "qwen35-vl-test",
        "--device",
        "mps",
        "--mmproj",
        str(mmproj),
        "--max-new-tokens",
        str(args.max_tokens),
    ]
    if args.expect_mtp_enabled:
        command.extend(
            [
                "--mtp",
                "--mtp-draft-tokens",
                str(args.mtp_draft_tokens),
                "--mtp-p-min",
                str(args.mtp_p_min),
            ]
        )

    success = False
    server = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base_url = f"http://{args.host}:{args.port}"
        wait_for_health(f"{base_url}/health", 15.0)

        if image is None:
            encoded = DEFAULT_IMAGE_BASE64
            mime_type = "image/png"
        else:
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            mime_type = image_mime(image)
        payload = {
            "model": "qwen35-vl-test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded}",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": args.max_tokens,
        }
        if args.temperature is not None:
            payload["temperature"] = args.temperature
        if args.expect_mtp_enabled:
            payload["mtp"] = True
            payload["mtp_draft_tokens"] = args.mtp_draft_tokens
            payload["mtp_p_min"] = args.mtp_p_min
        response, headers = post_json(
            f"{base_url}/v1/chat/completions",
            payload,
            args.timeout,
        )
        content = response["choices"][0]["message"].get("content", "").strip()
        if not content:
            raise RuntimeError(f"empty assistant content: {json.dumps(response)[:1000]}")
        if args.expect_mtp_disabled_reason:
            enabled = headers.get("x-kraken-mtp-enabled")
            reason = headers.get("x-kraken-mtp-disabled-reason")
            if enabled != "0" or reason != args.expect_mtp_disabled_reason:
                raise RuntimeError(
                    "unexpected VL MTP headers: "
                    f"enabled={enabled}, reason={reason}, headers={headers}"
                )
        if args.expect_mtp_enabled:
            enabled = headers.get("x-kraken-mtp-enabled")
            reason = headers.get("x-kraken-mtp-disabled-reason")
            if enabled != "1":
                raise RuntimeError(
                    "expected native VL MTP to be enabled: "
                    f"enabled={enabled}, reason={reason}, headers={headers}"
                )
        print(content)
        summary = {"usage": response.get("usage", {})}
        mtp_headers = {
            key: value for key, value in headers.items()
            if key.startswith("x-kraken-mtp-")
        }
        if mtp_headers:
            summary["mtp"] = mtp_headers
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        success = True
        return 0
    finally:
        server.terminate()
        try:
            stdout, _ = server.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            stdout, _ = server.communicate(timeout=5)
        if stdout and (not success or server.returncode not in {0, -15, -2, 1}):
            print(stdout, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
