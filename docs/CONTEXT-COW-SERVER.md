# Run the Context COW server on Linux

The server runs the model. BranchWrite runs on your other computer and sends
requests to this server. Start with one server and one GPU.

This preview is being checked on llm-60 with an RTX PRO 6000 Blackwell,
Qwen3.8-Flash-Next UD-Q4_K_XL, FP16 KV and a 98,304-token capacity.
Other machines and configurations need their own checks. This is not G1?G5
certification. The original mathematical gates remain separate.

## 1. Check the server computer

You need 64-bit Linux, an NVIDIA GPU, enough system RAM for your model, and
space for model files and the checkpoint pool. The client needs none of those
model files. Model downloads and model packing are separate preparation steps.

Run:

```sh
nvidia-smi
/usr/local/cuda/bin/nvcc --version
```

The first command lists the GPU. The second checks the CUDA compiler.
If either fails, fix the NVIDIA driver or CUDA toolkit before building.
This installer does not replace a working driver or change security settings.

## 2. Download and build

Install the small build tools on Ubuntu/Debian:

```sh
sudo apt update
sudo apt install -y git build-essential cmake python3-venv
```

Download the reviewed preview source. The exact tested ref will be recorded
when the clean server/client campaign finishes.

```sh
git clone https://github.com/CC-David-CC/Private_Context_COW.git
cd Private_Context_COW
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cow-server.txt
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release
cmake --build build --target strata-branch -j "$(nproc)"
```

A virtual environment is a separate folder for this server's Python packages.
The `.venv/bin/python` path selects that environment without an activation step.
The native build fetches its pinned GGML dependency. It does not need any old
research build directory or precompiled Context COW object file.

## 3. Choose the model files and storage

You need the complete model download and its matching Strata pack. Use the first
GGUF file if the download has several numbered parts. Keep all parts together.
The pack is a folder containing `dense.bin`, its index and tokenizer files.

Replace the two model paths below with your paths. Do not type the example paths
unless those folders exist on your computer.

```sh
.venv/bin/python tools/cow_server_setup.py --model /path/to/model-00001-of-00004.gguf --pack /path/to/pack --native build/strata-branch --output ~/cow-server --host 0.0.0.0 --port 8099 --ram-gib 48 --disk-gib 64
```

Setup hashes the real model files. This may take time for a large model.
It saves the model/build/settings identities, a private API key and the server
settings. It refuses to overwrite an existing server folder.

`--ram-gib` and `--disk-gib` are checkpoint budgets. Allow additional RAM and disk
space for the model, operating system, build and temporary work. Expired
checkpoint copies may be removed; retained canonical history remains available
for explicit reconstruction. Idle checkpoint expiry is 30 minutes in this
preview. No old saved computation is silently imported into a new profile.

## 4. Start and connect

```sh
.venv/bin/python tools/cow_server.py ~/cow-server
```

Leave this terminal open. A ready event means the API is listening.
Press Ctrl+C to close the owned server cleanly.

In a second terminal, show the key privately:

```sh
cat ~/cow-server/api-key.txt
hostname -I
```

Give your client the key and an address such as `http://10.0.7.68:8099`.
Use this computer's LAN address from `hostname -I`, not `127.0.0.1`.
Allow the chosen port only on your trusted LAN. This preview uses HTTP and a
bearer key; do not expose it to the public Internet.

On the client, follow BranchWrite's `docs/LINUX-QUICKSTART.md`.
The key is not saved by the client unless the user requests `--save-api`.

## Scope of this setup

This setup starts text and server-side GBNF/JSON Schema support. It does not
install a vision encoder, Blender, CAD or remote administration targets on the
client. Those capabilities require their separate configured boundaries.
Existing vision code and measurements remain in the repository.

Execution profiles bind the actual model, executable, source and settings.
Changing batch boundaries or operators is not promised to give the same logits.
This installation does not create a numerical oracle or grant certification.
Keep the source/runtime used by saved work; updating in place can invalidate it.
