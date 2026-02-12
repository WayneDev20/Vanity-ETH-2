# RunPod Setup — Vanity-ETH (4+4 prefix+suffix)

Step-by-step guide to run the vanity address generator on RunPod with GPU.

---

## Prerequisites

- RunPod account: [runpod.io](https://www.runpod.io/)
- Project folder (`Vanity-ETH 2`) on your computer
- Target Ethereum addresses (each must be 40 hex chars after `0x`)

---

## Step 1: Create a RunPod Pod

1. Go to [runpod.io/console/pods](https://www.runpod.io/console/pods)
2. Click **+ Deploy**, then **GPU Cloud**
3. Choose a GPU:
   - **RTX 4090** — ~$0.44/hr (recommended, ~3800 M/s)
   - **RTX 3090** — ~$0.34/hr (~1600 M/s)
   - **A100** — higher cost, very fast
4. Pick a **Template**:
   - `RunPod Pytorch 2.1` or any `*-cuda*-devel` image
   - Example: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`
5. Choose **80 GB** disk (or more)
6. Deploy and wait for the pod to start (status: **Running**)

---

## Step 2: Connect to the Pod

### Option A: Web Terminal

1. Click your pod in the RunPod console
2. Click **Connect**
3. Choose **Start Web Terminal**
4. You’ll get a terminal in your browser

### Option B: SSH

1. In the pod details, open **Connect** → **SSH**
2. Copy the SSH command (e.g. `ssh root@... -p 12345 -i ~/.ssh/id_rsa`)
3. Run it in your local terminal

---

## Step 3: Upload the Project

### Option A: RunPod File Browser

1. In the pod page, open **Connect** → **File Browser**
2. Go to `/workspace` or `/root`
3. Upload the `Vanity-ETH 2` folder (zip it first, then upload and extract)

### Option B: Git (if you use a repo)

```bash
cd /workspace   # or ~
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd Vanity-ETH-2   # or the folder name
```

### Option C: SCP (from your computer)

```bash
# From your local machine (replace with your pod details)
scp -P YOUR_PORT -r "/path/to/Vanity-ETH 2" root@YOUR_POD_IP:/workspace/
```

### Option D: Cloud Upload (Google Drive, Dropbox, etc.)

1. Upload the zip to the cloud
2. On the pod: `pip install gdown` (for Google Drive) or use `wget`/`curl`
3. Download and unzip into `/workspace` or `~`

---

## Step 4: One-Time Setup

1. Go to the project folder:

```bash
cd /workspace/Vanity-ETH-2
# or: cd ~/Vanity-ETH-2  (depending on where you uploaded)
```

2. Make scripts executable:

```bash
chmod +x runpod_setup.sh runpod_run.sh
chmod +x vanity-eth-address/runpod_build.sh
```

3. Run setup:

```bash
./runpod_setup.sh
```

This will:
- Check GPU (`nvidia-smi`)
- Build the CUDA binary (`vanity-eth-address`)
- Install Python deps (`ecdsa`, `pycryptodome`, `tqdm`)
- Create `addresses.txt` if it doesn’t exist

4. Confirm no errors and that a binary was built:

```bash
ls -la vanity-eth-address/vanity-eth-address
# Should show the executable
```

---

## Step 5: Add Target Addresses

Edit `addresses.txt` — one address per line:

```bash
# If nano is available:
nano addresses.txt

# If nano is not installed, use vim or vi:
vi addresses.txt
# (Press 'i' to edit, add your addresses, Esc then :wq to save)

# Or overwrite with echo:
echo "0x1234abcdef1234567890abcdef1234567890abcd" > addresses.txt
```

Example:

```
0x1234abcdef1234567890abcdef1234567890abcd
0xdead00000000000000000000000000000000beef
```

Rules:
- Each address must be 40 hex chars after `0x`
- The script matches the **first 4** and **last 4** hex chars
- Example: `0x1234...abcd` → prefix `1234`, suffix `abcd`

Save: `Ctrl+O`, Enter, then `Ctrl+X`.

---

## Step 6: Run

```bash
./runpod_run.sh
```

Or manually:

```bash
export VANITY_GPU=1
export VANITY_GPU_CUDA_BIN=$(pwd)/vanity-eth-address/vanity-eth-address
python3 vanity_address.py
```

---

## Step 7: Check Results

Results are written to `output.csv`:

```bash
cat output.csv
```

Columns:
- **Original Address** — from `addresses.txt`
- **New Address** — generated vanity address
- **Private Key** — private key for the new address

---

## Download Results

### Option A: Web file browser

1. Open the pod’s Connect → File Browser
2. Go to the project folder
3. Download `output.csv`

### Option B: SCP (from your computer)

```bash
scp -P YOUR_PORT root@YOUR_POD_IP:/workspace/Vanity-ETH-2/output.csv ./
```

---

## Optional: Environment Variables

| Variable | Purpose |
|----------|---------|
| `VANITY_GPU_CUDA_DEVICE` | GPU index (default `0`) for multi-GPU |
| `VANITY_GPU_TIMEOUT` | Max seconds per address (default `600`) |
| `VANITY_GPU_CUDA_WORK_SCALE` | Higher = more GPU work (e.g. `17`) |

Example:

```bash
export VANITY_GPU_CUDA_WORK_SCALE=17
./runpod_run.sh
```

---

## Full Command Sequence (Copy-Paste)

```bash
cd /workspace/Vanity-ETH-2
chmod +x runpod_setup.sh runpod_run.sh vanity-eth-address/runpod_build.sh
./runpod_setup.sh

# Edit addresses.txt, then:
./runpod_run.sh

# View results
cat output.csv
```

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| `nvcc not found` | Use a template with CUDA (e.g. `runpod/pytorch:*-cuda*-devel`) |
| `No such file: addresses.txt` | Create it with `nano addresses.txt` and add at least one address |
| `Nothing to do` | All addresses in `addresses.txt` are already in `output.csv` |
| Very slow | Increase `VANITY_GPU_CUDA_WORK_SCALE` (e.g. `17`) |
| Build fails | Ensure `nvidia-smi` works; try a different CUDA template |

---

## Time / Cost Estimate

| Pattern | RTX 4090 (~3800 M/s) |
|--------|----------------------|
| 4+4    | ~5–30 seconds        |
| 5+5    | ~15–30 minutes       |

For 4+4, RTX 4090 often finishes in seconds. Stop the pod when done to avoid extra charges.
