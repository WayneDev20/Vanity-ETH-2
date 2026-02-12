# RunPod Setup — Vanity-ETH (4+5 prefix+suffix)

Step-by-step guide to run the vanity address generator on RunPod with GPU.

---

## Quick Reference: Full Workflow

| Step | Action |
|------|--------|
| 1 | Terminate current pod (if any) |
| 2 | Deploy new pod (RTX 4090 + CUDA template) |
| 3 | Connect → Web Terminal |
| 4 | Clone from GitHub |
| 5 | Run `./runpod_setup.sh` |
| 6 | `nano addresses.txt` → add your addresses |
| 7 | `./runpod_run.sh` |
| 8 | Download `output.csv` when done |
| 9 | Terminate pod |

---

## Step 1: Terminate Your Current Pod

1. Go to [runpod.io/console/pods](https://www.runpod.io/console/pods)
2. Find your running pod in the list
3. Click the **three dots (⋮)** or **Actions** menu
4. Click **Terminate Pod** (or **Stop** then **Terminate**)
5. Confirm — the pod will be removed and billing stops

---

## Step 2: Create a New Pod

1. On [runpod.io/console/pods](https://www.runpod.io/console/pods), click **+ Deploy**
2. Choose **GPU Cloud**
3. Select a GPU:
   - **RTX 4090** — ~$0.44/hr (recommended, ~3800 M/s)
   - **RTX 3090** — ~$0.34/hr (~1600 M/s)
4. Select a **Template** with CUDA:
   - `RunPod Pytorch 2.1` or `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`
5. Set **Disk** to **80 GB** or more
6. Click **Deploy**
7. Wait until status is **Running**

---

## Step 3: Connect to the Pod

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

## Step 4: Clone the Project

### Option A: Git (recommended)

In the Web Terminal:

```bash
cd /workspace
git clone https://github.com/WayneDev20/Vanity-ETH-2.git
cd Vanity-ETH-2
```

### Option B: File Browser upload

1. In the pod page, open **Connect** → **File Browser**
2. Go to `/workspace`
3. Upload a zip of the project, then extract it

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

## Step 5: One-Time Setup

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

## Step 6: Add Target Addresses

Edit `addresses.txt` — one address per line:

```bash
nano addresses.txt
```

(`nano` is installed by `runpod_setup.sh`. If missing: `apt install nano`)

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

## Step 7: Run

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

## Step 8: Check Results

Results are written to `output.csv`:

```bash
cat output.csv
```

Columns:
- **Original Address** — from `addresses.txt`
- **New Address** — generated vanity address
- **Private Key** — private key for the new address

---

## Step 9: Download Results & Terminate

### Option A: Web file browser

1. Open the pod’s Connect → File Browser
2. Go to the project folder
3. Download `output.csv`

**Terminate pod (stop billing):** Pod → ⋮ menu → Terminate Pod → Confirm

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

After connecting to the pod Web Terminal:

```bash
cd /workspace
git clone https://github.com/WayneDev20/Vanity-ETH-2.git
cd Vanity-ETH-2
chmod +x runpod_setup.sh runpod_run.sh vanity-eth-address/runpod_build.sh
./runpod_setup.sh
nano addresses.txt
# Add your addresses, save (Ctrl+O, Enter, Ctrl+X)
./runpod_run.sh
cat output.csv
# Then terminate the pod in RunPod console
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
| 4+5    | ~15–60 seconds       |
| 5+5    | ~15–30 minutes       |

For 4+5, RTX 4090 often finishes in under a minute. Stop the pod when done to avoid extra charges.
