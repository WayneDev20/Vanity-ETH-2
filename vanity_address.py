import ecdsa
from Crypto.Hash import keccak
import csv
import os
import re
import subprocess
import shutil
from multiprocessing import Process, Manager, Pool, cpu_count
from tqdm import tqdm

# Match first N and last M hex chars of address (after 0x). 4+4 is faster than 5+5.
PREFIX_LEN = 4
SUFFIX_LEN = 4


def eth_address_from_pubkey(pubkey_bytes: bytes) -> str:
    # Ethereum address = last 20 bytes of Keccak-256(pubkey)
    k = keccak.new(digest_bits=256)
    k.update(pubkey_bytes)
    return "0x" + k.digest()[-20:].hex()


def _search_one_address(target_prefix: str, target_suffix: str):
    """Single-process search. Returns (address, priv_hex) or (None, None)."""
    curve = ecdsa.SECP256k1
    target_prefix = target_prefix.lower()
    target_suffix = target_suffix.lower()
    while True:
        sk = ecdsa.SigningKey.generate(curve=curve)
        vk = sk.verifying_key
        pubkey_bytes = vk.to_string()
        address = eth_address_from_pubkey(pubkey_bytes)
        addr_lower = address.lower()
        if addr_lower[2 : 2 + PREFIX_LEN] == target_prefix and addr_lower[-SUFFIX_LEN:] == target_suffix:
            return address, sk.to_string().hex()
        # else keep looping


def generate_matching_address(target_prefix, target_suffix, found_event, result):
    curve = ecdsa.SECP256k1
    target_prefix = target_prefix.lower()
    target_suffix = target_suffix.lower()

    while not found_event.is_set():
        sk = ecdsa.SigningKey.generate(curve=curve)
        vk = sk.verifying_key

        # Uncompressed public key = 64 bytes (x || y)
        pubkey_bytes = vk.to_string()

        address = eth_address_from_pubkey(pubkey_bytes)
        addr_lower = address.lower()

        # match first PREFIX_LEN and last SUFFIX_LEN hex chars (excluding 0x)
        if addr_lower[2 : 2 + PREFIX_LEN] == target_prefix and addr_lower[-SUFFIX_LEN:] == target_suffix:
            result["address"] = address
            result["priv"] = sk.to_string().hex()
            found_event.set()
            return


def find_vanity_address(original_address, num_processes=None):
    if not original_address.lower().startswith("0x"):
        print(f"Invalid address format: {original_address}")
        return None, None

    addr_lower = original_address.lower()[2:]
    if len(addr_lower) != 40:
        print(f"Invalid address length: {original_address}")
        return None, None

    prefix = addr_lower[:PREFIX_LEN]
    suffix = addr_lower[-SUFFIX_LEN:]

    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)

    if num_processes == 1:
        return _search_one_address(prefix, suffix)

    manager = Manager()
    found_event = manager.Event()
    result = manager.dict()

    processes = []
    for _ in range(num_processes):
        p = Process(
            target=generate_matching_address,
            args=(prefix, suffix, found_event, result),
        )
        p.start()
        processes.append(p)

    # Wait until one process finds a match
    found_event.wait()

    # Stop all workers
    for p in processes:
        p.terminate()
        p.join()

    if "address" in result:
        return result["address"], result["priv"]

    return None, None


def _find_vanity_gpu(original_address: str, gpu_bin: str, timeout_seconds=None):
    """
    Use an external GPU binary (e.g. profanity/vanity-eth-gpu) to find a matching address.
    Binary must support: --matching <40-char hex with X for wildcards> and print Private + Address to stdout.
    Returns (address, priv_hex) or (None, None).
    """
    if not original_address.lower().startswith("0x"):
        return None, None
    addr_lower = original_address.lower()[2:]
    if len(addr_lower) != 40:
        return None, None
    prefix, suffix = addr_lower[:PREFIX_LEN], addr_lower[-SUFFIX_LEN:]
    # Profanity-style: 40 chars, X = wildcard.
    pattern = prefix + "X" * (40 - PREFIX_LEN - SUFFIX_LEN) + suffix
    extra = os.environ.get("VANITY_GPU_EXTRA_ARGS", "").strip().split()
    cmd = [gpu_bin] + extra + ["--matching", pattern]
    try:
        cwd = os.path.dirname(os.path.abspath(gpu_bin)) or None
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_seconds,
            cwd=cwd if cwd else None,
        )
        # Decode with errors='replace' so GPU binary output (e.g. progress/binary chars) doesn't crash us
        stdout = (result.stdout or b"").decode("utf-8", errors="replace")
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        combined = stdout + "\n" + stderr
        # Parse: Private: 0x<64 hex> ... and Address: 0x<40 hex> (or "<name>: 0x<40>")
        priv_match = re.search(r"Private:\s*0x([0-9a-fA-F]{64})", combined)
        addr_match = re.search(r"0x([0-9a-fA-F]{40})", combined)
        # Prefer the 0x<40> that looks like an address (usually last or after "Private")
        if priv_match and addr_match:
            priv_hex = priv_match.group(1).lower()
            # Take last 40-char match as address (profanity prints address at end of line)
            addrs = re.findall(r"0x([0-9a-fA-F]{40})", combined)
            address = "0x" + (addrs[-1] if addrs else addr_match.group(1)).lower()
            return address, priv_hex
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None, None


def _parse_cuda_output(out: str):
    """Parse Private Key and Address from CUDA binary output (vanity-eth-address, etc.)."""
    priv_match = re.search(r"Private\s*Key:\s*0x([0-9a-fA-F]{64})", out)
    addr_match = re.search(r"Address:\s*0x([0-9a-fA-F]{40})", out)
    if not priv_match:
        priv_match = re.search(r"0x([0-9a-fA-F]{64})", out)
    if priv_match and addr_match:
        return "0x" + addr_match.group(1).lower(), priv_match.group(1).lower()
    return None, None


def _run_cuda_streaming(cmd: list, cwd: str, timeout_seconds: int) -> tuple:
    """
    Run CUDA binary with streaming output. vanity-eth-address prints results
    immediately but doesn't exit—we read stdout and return as soon as we get a match.
    """
    import threading
    import time

    output = []
    output_lock = threading.Lock()

    def reader(pipe):
        try:
            for line in iter(pipe.readline, b""):
                with output_lock:
                    output.append(line.decode("utf-8", errors="replace"))
        except Exception:
            pass

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd or None,
            bufsize=0,  # unbuffered to avoid line-buffering warning
        )
    except (FileNotFoundError, OSError):
        return None, None

    thread = threading.Thread(target=reader, args=(proc.stdout,))
    thread.daemon = True
    thread.start()

    start = time.time()
    while True:
        elapsed = time.time() - start
        if timeout_seconds and elapsed > timeout_seconds:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        with output_lock:
            out = "".join(output)
        addr, priv = _parse_cuda_output(out)
        if addr and priv:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            return addr, priv
        if proc.poll() is not None:
            break
        time.sleep(0.1)

    with output_lock:
        out = "".join(output)
    return _parse_cuda_output(out)


def _find_vanity_gpu_cuda(original_address: str, cuda_bin: str, timeout_seconds=None):
    """
    Use a CUDA vanity binary for GPU. Matches PREFIX + SUFFIX (e.g. 4+4 or 5+5 hex).
    Supports:
      1) vanity-eth-address: --prefix X --suffix Y --device N (doesn't exit, we stream)
      2) positional: <cuda_bin> <prefix> <suffix>
      3) flags: <cuda_bin> --device <N> --prefix <prefix> --suffix <suffix>
    Returns (address, priv_hex) or (None, None).
    """
    if not original_address.lower().startswith("0x"):
        return None, None
    addr_lower = original_address.lower()[2:]
    if len(addr_lower) != 40:
        return None, None
    prefix = addr_lower[:PREFIX_LEN]
    suffix = addr_lower[-SUFFIX_LEN:] if SUFFIX_LEN else ""
    cuda_device = os.environ.get("VANITY_GPU_CUDA_DEVICE", "0").strip() or "0"
    cuda_work_scale = os.environ.get("VANITY_GPU_CUDA_WORK_SCALE", "").strip()
    cwd = os.path.dirname(os.path.abspath(cuda_bin)) or None
    timeout = timeout_seconds or 600

    # vanity-eth-address style: --prefix X --suffix Y --device N (doesn't exit)
    cmd_vanity_eth = [cuda_bin, "--device", cuda_device, "--prefix", prefix]
    if suffix:
        cmd_vanity_eth.extend(["--suffix", suffix])
    if cuda_work_scale:
        cmd_vanity_eth.extend(["--work-scale", cuda_work_scale])

    # Try vanity-eth-address first (--prefix X --suffix Y; streams output, returns on first match)
    addr, priv = _run_cuda_streaming(cmd_vanity_eth, cwd, timeout)
    if addr and priv:
        return addr, priv

    # Style 2: positional args
    cmd_positional = [cuda_bin, prefix]
    if suffix:
        cmd_positional.append(suffix)
    addr, priv = _run_cuda_streaming(cmd_positional, cwd, timeout)
    if addr and priv:
        return addr, priv

    # Style 3: flags with --device
    cmd_flags = [cuda_bin, "--device", cuda_device, "--prefix", prefix]
    if suffix:
        cmd_flags.extend(["--suffix", suffix])
    if cuda_work_scale:
        cmd_flags.extend(["--work-scale", cuda_work_scale])
    addr, priv = _run_cuda_streaming(cmd_flags, cwd, timeout)
    if addr and priv:
        return addr, priv

    return None, None


def _worker_one_address(orig: str):
    """Used by Pool: find vanity for one address (single process). Returns (orig, new_addr, priv_key)."""
    new_addr, priv_key = find_vanity_address(orig, num_processes=1)
    return (orig, new_addr, priv_key)


# =========================
# Main script (batch-friendly: resume + flush)
# =========================

# Run this many addresses at once (each with 1 core). 0 = current behavior (one address, all cores).
SIMULTANEOUS_ADDRESSES = 8

# --- GPU (for rented GPU / cloud) ---
# Vast.ai note: OpenCL is often missing; prefer CUDA vanity miners (eth-wallet-888-cuda, vanity-eth-gpu-cuda, etc.).
# Environment knobs:
#   VANITY_GPU=1                 -> try GPU; auto-fallback to CPU unless VANITY_GPU_REQUIRE=1
#   VANITY_GPU_REQUIRE=1         -> exit with error if GPU binary not found/usable
#   VANITY_GPU_CUDA_BIN=/path    -> CUDA binary (prefix+suffix); preferred. Args: <prefix> [suffix]
#   VANITY_GPU_BIN=/path         -> OpenCL/profanity-style binary (fallback). Args: --matching <pattern>
#   VANITY_GPU_EXTRA_ARGS="..."  -> extra args for OpenCL binary
#   VANITY_GPU_TIMEOUT=600       -> kill hung GPU process after N seconds
USE_GPU = os.environ.get("VANITY_GPU", "").strip().lower() in ("1", "true", "yes")
GPU_REQUIRE = os.environ.get("VANITY_GPU_REQUIRE", "").strip().lower() in ("1", "true", "yes")
GPU_CUDA_BIN = os.environ.get("VANITY_GPU_CUDA_BIN", "").strip()  # CUDA binary (prefix+suffix); preferred if set
GPU_BIN = os.environ.get("VANITY_GPU_BIN", "").strip() or "profanity"
GPU_TIMEOUT = int(os.environ.get("VANITY_GPU_TIMEOUT", "0") or "0") or None


def _resolve_gpu_binaries():
    """Resolve CUDA/OpenCL vanity binaries and give clear errors (esp. on Vast.ai)."""
    cuda_path = None
    opencl_path = None

    if GPU_CUDA_BIN:
        # Prefer absolute path if file exists, otherwise try PATH.
        if os.path.isfile(GPU_CUDA_BIN):
            cuda_path = os.path.abspath(GPU_CUDA_BIN)
        else:
            cuda_path = shutil.which(GPU_CUDA_BIN)

    if not cuda_path and GPU_BIN:
        if os.path.isfile(GPU_BIN):
            opencl_path = os.path.abspath(GPU_BIN)
        else:
            opencl_path = shutil.which(GPU_BIN)

    return cuda_path, opencl_path


def _ensure_gpu_ready():
    """Run quick sanity checks so we fail fast instead of silently falling back to CPU."""
    if not USE_GPU:
        return None, None

    cuda_path, opencl_path = _resolve_gpu_binaries()

    if not (cuda_path or opencl_path):
        msg = (
            "GPU enabled but no vanity binary found. "
            "Set VANITY_GPU_CUDA_BIN to your CUDA miner (recommended on Vast.ai) "
            "or VANITY_GPU_BIN to an OpenCL/profanity binary."
        )
        if GPU_REQUIRE:
            raise SystemExit(msg)
        print(msg + " Falling back to CPU.")
        return None, None

    # Optional: check that nvidia-smi works to catch broken drivers early.
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        if GPU_REQUIRE:
            raise SystemExit("GPU required but nvidia-smi not available; CUDA drivers may be missing.")
        print("Warning: nvidia-smi not available; continuing anyway.")

    return cuda_path, opencl_path

input_file = "addresses.txt"   # one address per line
output_file = "output.csv"    # results

if __name__ == "__main__":
    # Load all input addresses (handles hundreds to thousands)
    # Skip empty lines and lines starting with #
    with open(input_file, "r") as f:
        all_addresses = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    # Resume: skip addresses already in output.csv
    completed = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and header[0] == "Original Address":
                    for row in reader:
                        if row and row[0].strip():
                            completed.add(row[0].strip().lower())
        except (csv.Error, OSError):
            pass

    original_addresses = [a for a in all_addresses if a.strip().lower() not in completed]
    skipped = len(all_addresses) - len(original_addresses)

    if skipped:
        print(f"Resuming: {skipped} address(es) already in {output_file}, {len(original_addresses)} remaining.")
    if not original_addresses:
        print("Nothing to do. All addresses already processed.")
        exit(0)

    # Resolve GPU binaries and sanity check environment (fast-fail on Vast.ai)
    gpu_cuda_bin_resolved, gpu_bin_resolved = _ensure_gpu_ready()
    if gpu_cuda_bin_resolved:
        print(f"Using GPU (CUDA, prefix+suffix): {gpu_cuda_bin_resolved}")
    elif gpu_bin_resolved:
        print(f"Using GPU binary (OpenCL): {gpu_bin_resolved}")
    else:
        USE_GPU = False

    # Open output: append if resuming, write new if not
    is_new_file = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    mode = "w" if is_new_file else "a"
    with open(output_file, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if is_new_file:
            writer.writerow(["Original Address", "New Address", "Private Key"])
            csvfile.flush()

        if USE_GPU and (gpu_cuda_bin_resolved or gpu_bin_resolved):
            # GPU: one address at a time
            for orig in tqdm(original_addresses, desc="Processing (GPU)", unit="addr"):
                if gpu_cuda_bin_resolved:
                    new_addr, priv_key = _find_vanity_gpu_cuda(orig, gpu_cuda_bin_resolved, GPU_TIMEOUT)
                else:
                    new_addr, priv_key = _find_vanity_gpu(orig, gpu_bin_resolved, GPU_TIMEOUT)
                if not new_addr or not priv_key:
                    # GPU failed; optionally force-stop if user required GPU
                    if GPU_REQUIRE:
                        writer.writerow([orig, "GPU failed", "N/A"])
                        csvfile.flush()
                        raise SystemExit("GPU required but vanity binary failed for an address; see prior logs.")
                    new_addr, priv_key = find_vanity_address(orig)
                if new_addr:
                    writer.writerow([orig, new_addr, priv_key])
                else:
                    writer.writerow([orig, "Failed to generate", "N/A"])
                csvfile.flush()
        elif SIMULTANEOUS_ADDRESSES > 0:
            # CPU: Run N addresses at once (1 core each).
            n_workers = min(SIMULTANEOUS_ADDRESSES, len(original_addresses), max(1, cpu_count()))
            print(f"Running {n_workers} address{'es' if n_workers != 1 else ''} simultaneously (1 core each).")
            with Pool(processes=n_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_worker_one_address, original_addresses, chunksize=1),
                    total=len(original_addresses),
                    desc="Processing",
                    unit="addr",
                ):
                    orig, new_addr, priv_key = result
                    if new_addr:
                        writer.writerow([orig, new_addr, priv_key])
                    else:
                        writer.writerow([orig, "Failed to generate", "N/A"])
                    csvfile.flush()
        else:
            # CPU: one address at a time, all cores for that address.
            for orig in tqdm(original_addresses, desc="Processing", unit="addr"):
                new_addr, priv_key = find_vanity_address(orig)
                if new_addr:
                    writer.writerow([orig, new_addr, priv_key])
                else:
                    writer.writerow([orig, "Failed to generate", "N/A"])
                csvfile.flush()

    print(
        f"Done. {len(original_addresses)} address(es) processed. Results in {output_file}.\n"
        "Note: This is brute force and can take a long time depending on how rare the pattern is."
    )
