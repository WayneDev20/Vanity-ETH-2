import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path

import runpod

APP_ROOT = Path("/workspace/Vanity-ETH-2")
VANITY_SCRIPT = APP_ROOT / "vanity_address.py"
VANITY_BIN = APP_ROOT / "vanity-eth-address" / "vanity-eth-address"
ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


def _normalize_addresses(raw_value):
    if isinstance(raw_value, str):
        candidates = [line.strip() for line in raw_value.splitlines()]
    elif isinstance(raw_value, list):
        candidates = [str(item).strip() for item in raw_value]
    else:
        raise ValueError("input.addresses must be a string or list")

    addresses = []
    invalid = []
    for candidate in candidates:
        if not candidate or candidate.startswith("#"):
            continue
        if ADDRESS_RE.match(candidate):
            addresses.append(candidate)
        else:
            invalid.append(candidate)

    if not addresses:
        raise ValueError("No valid addresses were provided.")

    if invalid:
        raise ValueError(f"Invalid addresses: {invalid[:10]}")

    return addresses


def _run_vanity_batch(addresses, timeout_seconds):
    env = os.environ.copy()
    env.setdefault("VANITY_GPU", "1")
    env.setdefault("VANITY_GPU_CUDA_BIN", str(VANITY_BIN))
    env.setdefault("VANITY_GPU_DELAY_SEC", "1")
    env.setdefault("VANITY_GPU_RETRIES", "2")

    with tempfile.TemporaryDirectory(prefix="vanity-job-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "addresses.txt"
        output_path = temp_path / "output.csv"

        input_path.write_text("\n".join(addresses) + "\n", encoding="utf-8")

        result = subprocess.run(
            ["python3", str(VANITY_SCRIPT)],
            cwd=temp_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-2000:]
            stdout_tail = (result.stdout or "")[-2000:]
            raise RuntimeError(
                "Vanity job failed. "
                f"stdout_tail={stdout_tail!r} stderr_tail={stderr_tail!r}"
            )

        if not output_path.exists():
            raise RuntimeError("Vanity job completed but output.csv was not generated.")

        rows = []
        with output_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                rows.append(
                    {
                        "original_address": row.get("Original Address", ""),
                        "new_address": row.get("New Address", ""),
                        "private_key": row.get("Private Key", ""),
                    }
                )

        csv_text = output_path.read_text(encoding="utf-8")
        return rows, csv_text


def handler(job):
    job_input = job.get("input", {}) or {}

    if not VANITY_SCRIPT.exists():
        return {"error": f"Missing script: {VANITY_SCRIPT}"}
    if not VANITY_BIN.exists():
        return {"error": f"Missing CUDA binary: {VANITY_BIN}"}

    try:
        addresses = _normalize_addresses(job_input.get("addresses", ""))
    except ValueError as error:
        return {"error": str(error)}

    try:
        timeout_seconds = int(job_input.get("timeout_seconds", 7200))
    except (TypeError, ValueError):
        return {"error": "input.timeout_seconds must be an integer."}

    if timeout_seconds < 60:
        return {"error": "input.timeout_seconds must be at least 60."}

    try:
        rows, csv_text = _run_vanity_batch(addresses, timeout_seconds)
    except subprocess.TimeoutExpired:
        return {"error": f"Job timed out after {timeout_seconds} seconds."}
    except Exception as error:  # noqa: BLE001
        return {"error": str(error)}

    return {
        "count": len(rows),
        "rows": rows,
        "csv": csv_text,
    }


runpod.serverless.start({"handler": handler})
