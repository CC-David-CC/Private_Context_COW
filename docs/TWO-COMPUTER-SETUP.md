# Put Context COW on your desktop and BranchWrite on your laptop

You can use two Linux computers on the same home network:

- **Desktop / server:** runs the AI model and keeps its saved context.
- **Laptop / client:** runs BranchWrite and holds the files you want to change.

You do not need an AI agent to follow these steps. You need Internet access for
installation. Read the hardware and download size before starting.

## Before you download

This guide uses the Q4 model tested with Context COW. It is a large model:
**about 111.3 GB to download**, plus about 1.4 GiB for its prepared model pack.
Allow **200 GB of free disk space** for the model, build and default disk cache.
Use a fast SSD. The laptop does not download the model.

The tested server has Linux, **128 GB of system RAM** and an NVIDIA RTX PRO 6000
Blackwell with **96 GB of graphics memory**. This describes the tested machine,
not a claim that every gaming PC must have that card or that every smaller card
will work. A usual 16/32 GB RAM gaming PC is not this tested Q4 setup. Other GPU
and memory combinations still need their own check.

These copy-and-paste instructions target 64-bit Intel/AMD Ubuntu 24.04 or newer.
The server needs a working NVIDIA driver and the CUDA Toolkit. CUDA is NVIDIA's
software for building and running GPU programs. The installer does not replace
your graphics driver or install the CUDA compiler automatically.

## Part A — on the desktop

### 1. Check NVIDIA and CUDA

On Ubuntu, press Ctrl+Alt+T to open Terminal on the desktop. Run these commands one at a time:

```sh
nvidia-smi
/usr/local/cuda/bin/nvcc --version
free -h
df -h "$HOME"
```

The first command should show your NVIDIA card. The second should show a CUDA
compiler version. The last two show RAM and free disk space. If the first or
second command fails, fix the NVIDIA/CUDA installation before continuing.
Do not start the model download hoping that it will install the driver.

A working graphics driver does not necessarily include the CUDA compiler.
For a missing toolkit, follow [NVIDIA's Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
and [CUDA download page](https://developer.nvidia.com/cuda-downloads).
Select your Linux distribution and follow its toolkit instructions. The measured
server build used CUDA 13.2. This guide does not automatically change your driver.

### 2. Install the small build tools

```sh
sudo apt update
sudo apt install -y git curl ca-certificates build-essential cmake python3-venv
```

`sudo` lets Linux install system packages. Type your Linux password if asked.
You will not see letters while you type the password. That is normal.

### 3. Get Context COW and its Python packages

```sh
mkdir -p ~/Apps
cd ~/Apps
git clone --branch lan-preview-2026-10-20.1 https://github.com/CC-David-CC/Private_Context_COW.git ContextCOW
cd ContextCOW
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cow-server.txt
```

A **virtual environment** is a separate folder for this app's Python packages.
You do not need to activate it. Commands beginning with `.venv/bin/python` use it.
The repository name contains “Private”, but this is the public download URL.

### 4. Build the COW engine

```sh
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release
cmake --build build --target strata-branch -j "$(nproc)"
```

Wait for the build to finish without errors. It downloads the pinned native build
dependency. It builds the COW engine from this checkout, not an ordinary Strata
server downloaded from elsewhere.

### 5. Download and prepare the model automatically

```sh
.venv/bin/python tools/cow_prepare_model.py --yes
```

This command uses the existing Strata downloader and native model packer. It:

1. Downloads the four pinned Q4 files into `~/ContextCOW-data/models/UD-Q4_K_XL`.
2. Checks each file against its expected size and SHA-256 hash. A hash is a
   fingerprint used to detect a wrong or damaged file.
3. Gets the pinned GGUF packing tools.
4. Builds the matching pack in `~/ContextCOW-data/pack-ud-q4-k-xl`.
5. Checks the prepared pack and prints **Model preparation complete**.

It does not download MTP or start a model server. `--yes` approves the displayed
large download. If the connection drops, run the same command again. Downloads
resume; finished model files are still checked. Hash checks can take several
minutes. Progress is printed during a long hash check.

### 6. Make your server settings and API key

Copy this whole command. The home-folder paths work without replacing a username:

```sh
.venv/bin/python tools/cow_server_setup.py --model "$HOME/ContextCOW-data/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" --pack "$HOME/ContextCOW-data/pack-ud-q4-k-xl" --native build/strata-branch --expert-profile data/expert-profile.bin --output "$HOME/cow-server" --host 0.0.0.0 --port 8099 --ram-gib 48 --disk-gib 64
```

This creates a new server folder and a random API key. The key is a password for
this service. Setup hashes the model and runtime files; allow time for that.
These pool sizes belong to the reference server above. Setup will not replace an
existing `~/cow-server` folder. To start it again later, use step 7, not step 6.

The **RAM pool** keeps reusable context in memory. The **disk pool** keeps saved
computational checkpoints. Their 48/64 GiB limits are additional budgets, not the
size of the model. Expiring a checkpoint may require computation to be repeated;
it does not turn your saved conversation into an empty chat.

### 7. Start the server

```sh
.venv/bin/python tools/cow_server.py "$HOME/cow-server"
```

Keep this terminal open. Wait for the ready message. Open a second desktop
terminal and run:

```sh
hostname -I
cat "$HOME/cow-server/api-key.txt"
```

Write down the desktop's LAN IP address and copy its key privately. For example,
if the address is `192.168.1.50`, the laptop will use:

```text
http://192.168.1.50:8099
```

Use your own address, not that example. `0.0.0.0` is a server listening setting;
it is not the address to type on the laptop. `127.0.0.1` means “this computer”,
so it would point at the laptop itself.

Keep this HTTP service on your trusted LAN. Do not forward port 8099 through your
router to the Internet. Only project content sent to this server is used by it;
choose a server you trust.

## Part B — on the laptop

### 8. Install BranchWrite

Open Terminal on the laptop. Install its small system tools:

```sh
sudo apt update
sudo apt install -y curl git ca-certificates bubblewrap util-linux
```

Then install BranchWrite as your normal user, without `sudo`:

```sh
curl -fsSL https://raw.githubusercontent.com/CC-David-CC/BranchWrite/lan-preview-2026-10-20.1/install.sh -o "$HOME/branchwrite-install.sh" && sh "$HOME/branchwrite-install.sh"
```

Wait for **BranchWrite is installed**. It downloads its own Python and client
packages. It does not install a GPU engine or download the model onto the laptop.

### 9. Open a project folder and connect

```sh
mkdir -p ~/my-first-website
cd ~/my-first-website
~/.local/bin/bw
```

At **Server address**, type the desktop address from step 7, including `http://`
and `:8099`. At **API key**, paste the desktop's key and press Enter. The key
stays in memory for this session. It is not saved unless you choose `--save-api`.

BW works on the folder where you started it. It can read, create and change the
project files. Keep passwords and private keys outside this project folder.

### 10. Make something and change your mind

Type:

> Make a pretty website about dogs. Make it work on a phone. Add a button that
> shows a friendly greeting. Keep everything in one file and do not download
> anything.

When BW finishes, type `/preview` to view its saved website. Then ask:

> Make the dog names bold. Keep the greeting button working.

You do not need to know HTML or choose exact file names. Work mode makes the
changes and keeps saved file versions. Open the same folder and run BW again to
return to its saved session.

### 11. Talk while it works

- **Enter:** send a request or add new information while it is working.
- **Tab:** queue a note for the next turn. Queued follow-ups are combined.
- **Esc:** ask it to stop at a safe boundary.
- **Ctrl+R:** search your earlier input.
- **/plan:** discuss a change without editing files.
- **/work:** return to editing.
- **/aside your question:** ask a side question without changing the main tip.
- **/exit:** save and leave.

The browser worker has its own optional setup. The standard client installer does
not yet enable automatic screenshot-driven website inspection. `/preview` lets
you inspect the result in your own browser.

## If the laptop cannot connect

1. Check that the desktop server terminal is still open and says ready.
2. Check the address and port. Do not use `127.0.0.1` for another computer.
3. Put both computers on the same LAN; guest Wi-Fi may block other devices.
4. Copy the key again without adding spaces or quotation marks.
5. If a desktop firewall is enabled, allow TCP port 8099 from the laptop's LAN
   address. Do not open the port to the entire Internet.

If a download stops, rerun step 5. If a hash fails, the preparation command names
the bad file and stops; it does not silently accept it. If a build or model load
fails, keep its error text. “Setup installed” and “the model is running” are
different results.

## Stop and return tomorrow

Exit BW with `/exit`. On the desktop, press Ctrl+C in the server terminal.
To return, open the desktop terminal and run:

```sh
cd ~/Apps/ContextCOW
.venv/bin/python tools/cow_server.py "$HOME/cow-server"
```

Then start `~/.local/bin/bw` from the same laptop project folder. Do not delete the
server folder or its source checkout to update in place while saved work uses it.

## What has actually been tested

A fresh Linux client installation, connection to a real Q4 server, website creation,
follow-up edits and resuming after a server-process restart have passed bounded
checks. The resource-preparation helper has separate download-resume, hash and
pack-publication tests. Its real Q4 pack check is reported in the setup evidence.
The complete 111 GB Internet download has not been repeated for this guide.
A different gaming PC still needs its own build, load and capability checks.
This guide does not establish full G1–G5 numerical or release certification.
