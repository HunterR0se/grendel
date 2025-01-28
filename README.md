Grendel

```bash
  ________                          .___     .__           🛠️ Build 158 🛠️
 /  _____/______   ____   ____    __| _/____ |  |  🕹️ CUDA Acceleration 🕹️
/   \  __\_  __ \_/ __ \ /    \  / __ |/ __ \|  |  Updated Jan 26, 2025 ⏱️
\    \_\  \  | \/\  ___/|   |  \/ /_/ \  ___/|  |__
 \________/__|    \____/\___|__/\_____|\____/_____/   🌹 by Hunter Rose 🌹
```

**Author:** [Hunter Rose](https://x.com/HunterR0se)

## Overview

Grendel is a tool designed to generate and test cryptographic keys against known Bitcoin addresses. The project supports both CPU and NVIDIA GPU acceleration for key generation.

Key features include:

- Detection of CPU or NVIDIA GPU availability on startup.
- Integration with LevelDB for storing known addresses.
- Periodic loading of new Bitcoin blocks during runtime.
- Logging of found addresses with their seeds, private keys, and balances.

### Errata

Yes, I am aware the code is extensive and needs to be cleaned up. Feel free to fork, contribute, or offer suggestions. There are some functions that are far too complicated and some functions that can be removed, as they are no longer used.

The current speed of ~1.8MM generated addresses per second (including full address matching) on an RTX 4090 is impressive, but I do believe we can improve this speed. Open Source ftw.

## Getting Started

If you want to short-circuit all the compilation and everything else, here's the latest binary for Linux. It will not (and should not) run on Windoze.

### Example Startup & Loading

The system provides comprehensive and detailed logs of each stage, including address generation, matching, and dropped or found addresses.

```bash
[🔍 -INFO] ─────────────────────────────────────────────────────────────────
[🔍 -INFO] Generator: Enabled Track All: Enabled
[🔍 -INFO] Reparse: Disabled GPU Mode: Enabled
[🔍 -INFO] DebugMode: Disabled CPU Mode: Disabled
[🎮 -GPU-] ─────────────────────────────────────────────────────────────────
[🎮 -GPU-] CPU vs GPU Benchmark (5,000,000 addresses)
[🎮 -GPU-] ─────────────────────────────────────────────────────────────────
[🎮 -GPU-] CPU: 16.381225411s (305,227 keys/sec)
[🎮 -GPU-] GPU: 701.274596ms (7,129,874 keys/sec)
[🎮 -GPU-] Speedup: 23.36x
[🎮 -GPU-] ─────────────────────────────────────────────────────────────────
[🎮 -GPU-] System has 256 Cores and 503.6 GB RAM
[🎮 -GPU-] NVIDIA GeForce RTX 4090 23GB VRAM (CUDA Enabled)
[🎮 -GPU-] ─────────────────────────────────────────────────────────────────
[🔍  INFO] Loading Addresses: 2.34GB
[🔍  INFO] Total Addresses:   55,006,296 (151.6 seconds)
[📁 -DATA] ─────────────────────────────────────────────────────────────────
[📁 -DATA] Starting the Block Loader...
[📁 -DATA] ─────────────────────────────────────────────────────────────────
[📁 -DATA] Block loader successfully started.
[🔍  INFO] ─────────────────────────────────────────────────────────────────
[🔍  INFO] Checked 5,000,000 addresses in 8.783 seconds
[🔍  INFO] Matching rate: 569,277 addresses/second
[🧠 -MEM-] ─────────────────────────────────────────────────────────────────
[🧠 -MEM-] CPU Cores:  256 Disk:   120.0GB Free:    91.4GB
[🧠 -MEM-] Total:  503.6GB Avail:  336.0GB Write:  512.0GB
[🧠 -MEM-] Block:  334.9GB Write:   95.7GB Comp:    47.8GB
[🧠 -MEM-] Batch:     500k Chan:     1000k RNG:      1048k
[〰️ HEADR] ─────────────────────────────────────────────────────────────────
[📝 STATS] Total Addresses:  54,526,722
[📝 STATS]      Legacy (1):  18,980,348
[📝 STATS]      SegWit (3):  18,213,445
[📝 STATS]    Native (bc1):  17,332,929
[📝 STATS] ─────────────────────────────────────────────────────────────────
[🔍  INFO] Starting generator in GPU Enabled mode
[🔍  INFO] Worker Pool - 512 (workers) 8,388,608 (buffer)
[〰️ HEADR] ─────────────────────────────────────────────────────────────────
[〰️ HEADR]  GENERATED |  RATE/s | LEGACY  |  SEGWIT |  NATIVE |  RAM  | DROP
[〰️ HEADR] ─────────────────────────────────────────────────────────────────
[16:00:59]       107M | 1785.8k | 16.803M | 16.809M | 17.388M | 14.2G | 0
[16:01:59]       211M | 1732.5k | 17.131M | 17.139M | 17.729M | 11.9G | 0
[16:02:59]       314M | 1720.3k | 16.965M | 16.975M | 17.560M | 10.6G | 0
[16:03:59]       431M | 1948.1k | 19.272M | 19.282M | 19.946M | 11.5G | 0
```

- Note the efficient use of RAM, the different address types being generated and matched, the lack of dropped addresses being matched, and the consistency of the number of generated and matched addresses -- about ~1.7MM per second, and even more as the system ramps up and the cache becomes stable.

## Important Notice

This tool was developed as a technical demonstration to illustrate several important concepts:

1. The incredible security of Bitcoin's cryptographic foundations

2. The practical impossibility of randomly generating private keys that match existing Bitcoin addresses

3. The implementation of high-performance computing techniques, including GPU acceleration and parallel processing

As highlighted in the conclusion, even with state-of-the-art hardware (RTX 4090) generating 1.8 million addresses per second, it would take over 120 years to have a reasonable chance of finding a single valid address. This project serves as a practical demonstration of why Bitcoin's cryptographic security is robust against brute-force attacks.

The code base also showcases modern software engineering practices including:

- CUDA GPU optimization
- Efficient database management
- Real-time blockchain data processing
- Performance profiling and optimization
- Cross-platform systems programming

This project is intended for educational purposes and to demonstrate technical proficiency in cryptography, distributed systems, and high-performance computing. It is not designed or intended as a hacking tool, as the mathematics behind Bitcoin's security make such attempts futile.

Researchers and developers interested in blockchain technology, cryptography, or high-performance computing may find this codebase useful for understanding these concepts in practice.

### Download Binary

1. **Download the Pre-compiled Binary:**

    ```bash
    wget https://github.com/HunterRose42/grendel/blob/main/bin/grendel -O grendel
    wget https://github.com/HunterRose42/grendel/blob/main/bin/libkeygen.so -O grendel
    ```

2. **Make it Executable and Move to System Path:**

    ```bash
    chmod +x bin/grendel
    sudo mv bin/grendel /usr/local/bin/
    sudo mv bin/libkeygen.so /usr/local/lib
    ```

3. **Verify Installation:**

    ```bash
    grendel
    ```

4. **Ensure Address Files:**
    - Make sure your Bitcoin address files are in the `~/.bitcoin` directory
    - If not present, run Bitcoin Core first to sync the blockchain

> **Note:** The binary is compiled for x86_64 Linux systems. For other architectures, please build from source.

### Command Line Options

```bash
[⌛️ START] ────────────────────────────────────────────────────────
[⌛️ START] ₿ Grendel Commands:
[⌛️ START] --debug     : Enable Debug Mode
[⌛️ START] --gen       : Generate Addresses (true by default)
[⌛️ START] --import    : Re-import all addresses
[⌛️ START] --track-all : Track all addresses (uses more memory)
[⌛️ START] --cpu       : Force CPU mode for testing
[⌛️ START] --gpu       : Force GPU mode for testing
[⌛️ START] --bench     : Run Benchmark and exit
[⌛️ START] ───────────────────────────────────────────────────────
```

#### Profiling

If running with `--profile` enabled (not shown in the list), the binary will create a `cpu_profile.pprof` and `/gpu_profile.pprof` file, which can be checked with go's pprof profiling.

Example:
`go tool pprof cpu_profile.pprof `

Typically, `top 10` is the best place to start.

### Prerequisites

- A Linux-based operating system (e.g., Debian, Arch, Ubuntu)
- Bitcoin Core installed and synced (see Installation below)

### Installation

#### System Setup

1. **Update System:**

    ```bash
    sudo apt update
    ```

2. **Install Basic Tools:**

    ```bash
    sudo apt install git vim zsh tmux
    ```

3. **Configure Zsh as Default Shell:**

    ```bash
    chsh -s $(which zsh)
    ```

4. **Install Oh-My-Zsh:**

    ```bash
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
    ```

5. **Set Time Zone:**

a. First, let's update the system timezone database:

```bash
sudo apt-get update
sudo apt-get install tzdata
```

b. Then set the timezone:

```bash
echo "America/New_York" | sudo tee /etc/timezone
sudo dpkg-reconfigure -f noninteractive tzdata
```

#### Bitcoin Core Installation

1. **Install Dependencies:**

    ```bash
    sudo apt install apt-transport-https curl gnupg wget tmux htop
    ```

2. **Download Bitcoin Core:**

    ```bash
    mkdir install
    cd install
    wget https://bitcoincore.org/bin/bitcoin-core-28.0/bitcoin-28.0-x86_64-linux-gnu.tar.gz
    ```

3. **Extract and Install Bitcoin Core:**

    ```bash
    tar zxvf bitcoin-28.0-x86_64-linux-gnu.tar.gz
    sudo install -m 0755 -o root -g root -t /usr/local/bin bitcoin-28.0/bin/*
    ```

4. **Cleanup:**

    ```bash
    cd ~
    rm -rf install
    ```

5. **Configure Bitcoin:**

    ```bash
    mkdir ~/.bitcoin
    cp <project>/bitcoin/bitcoin.conf ~/.bitcoin/bitcoin.conf
    ```

6. **Start Bitcoin Daemon:**

    ```bash
    tmux new -s Bitcoin
    bitcoind
    ```

    Press `CTRL+B` then `D` to detach from the tmux session. Monitor progress using:

    ```bash
    bitcoin-cli info
    ```

### Grendel Setup

1. **Compile and Deploy Grendel:**

    - Modify `~/.ssh/config` script and add your remote server credentials.
    - Ensure your login uses SSH without requiring a password for the best experience.
    - Run `./bitbuild` to compile the code and push it to the SSH server's `/usr/local/bin` directory.

    ```bash
    bin/bitbuild build <server>
    ```

2. **Start Grendel:**

    - SSH into the server and run:

        ```bash
        grendel --import
        ```

    - Best practice is to run in a tmux terminal:

        ```bash
        tmux new -s Grendel
        grendel run
        ```

        You can then exit with `CTRL+B` then `D` and return to the tmux session with `tmux a -t Grendel` to monitor the application.

## tmux Color Issues

1. Ensure your tmux is recent enough (3.2+):

    ```bash
    tmux -V
    ```

2. Add these specific settings to your `~/.tmux.conf`:

    ```bash
    # Enable UTF-8
    set -q -g status-utf8 on
    setw -q -g utf8 on

    # Terminal settings
    set -g default-terminal "tmux-256color"
    set -as terminal-features ",xterm-256color:RGB"
    set -as terminal-overrides ',*:Smulx=\E[4::%p1%dm'
    set -as terminal-overrides ',*:Setulc=\E[58::2::%p1%{65536}%/%d::%p1%{256}%/%{255}%&%d::%p1%{255}%&%d%;m'
    ```

3. Install necessary fonts and set them in your terminal:

    ```bash
    sudo apt install fonts-noto-color-emoji fonts-powerline
    ```

4. Add to your `~/.bashrc` or `~/.zshrc`:

    ```bash
    export LANG=en_US.UTF-8
    export LC_ALL=en_US.UTF-8
    export TERM=xterm-256color
    ```

5. After making these changes:

    ```bash
    # Reload tmux config if tmux is running
    tmux source-file ~/.tmux.conf

    # Restart your shell
    source ~/.zshrc
    ```

### Usage

- On startup, Grendel checks for the availability of a GPU; otherwise, it defaults to CPU usage.
- If `addresses.db` does not exist, all addresses will be imported into Grendel from the Bitcoin node block files (in .bitcoin) on startup.
- For a smaller, more lightweight set of addresses, consider the `--track-all` option to limit the total number of addresses on lower-powered systems (under 8 CPUs).

#### Operation Flow

1. Load addresses from LevelDB.
2. Load all the `.bitcoin` block addresses.
3. Generate unique keys (using GPU if available, otherwise CPU).
4. Test generated keys against known addresses.
5. Periodically load additional blocks during runtime.
6. If an address is found:
    - Log it on screen.
    - Write to `found.json` with the Seed, Private Key, and the address & balance.

#### Optional NVIDIA Driver Installation (for GPU Support)

1. **Update System and Install Dependencies:**

    ```bash
    sudo apt update
    sudo apt install software-properties-common
    ```

2. **Add Graphics Drivers PPA:**

    ```bash
    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt update
    ```

3. **Install and Auto-Install NVIDIA Drivers:**

    ```bash
    sudo apt install ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    ```

## Bonus (Coming Soon!)

    **The entire f\*cking blockchain of addresses!**

    ```bash
    # create the directory
    mkdir -p .config
    ```

    You MUST install ipfs to do this;

    ```bash
    wget https://dist.ipfs.tech/kubo/v0.32.1/kubo_v0.32.1_linux-amd64.tar.gz
    tar -xvzf kubo_v0.32.1_linux-amd64.tar.gz
    cd kubo
    sudo bash install.sh
    ```

    Now, get the full list, making sure you are in the correct directory first.

    ```bash
    cd ~/.config
    ipfs get QmbxzMK853WF4qd9Wmbt3tUKu36eSqN5E7kqot1ft5WPmT -o addresses.txt.gz
    ```

    This retrieves the entire 1.8GB+ address file from IPFS. Once downloaded, run;

    ```bash
    grendel --import
    ```

    And this will create the full LevelDB of all addresses from the Blockchain (over 55 million total addresses with balances).

    - Note: The import process is very efficient. Importing 55 million addresses will happen quickly, with logs, and will take ~3GB of space. Once done, you can discard the original addresses.txt.gz file, as the binary uses the database in `.bitcoin/addresses.db` after the import is complete.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with your changes.

Hat tip to;
https://github.com/Harold-Glitch/bitreverse/blob/master/secp256k1.cuh
https://github.com/XopMC/CudaBrainSecp/tree/main/GPU

### Real-World Consideration

The actual probability of randomly generating a key that matches an existing address with a non-zero balance is extremely low, perhaps on the order of 1 in 10^12 or even lower, due to the vast number of possible keys and the relatively small number of addresses with non-zero balances.

#### Conclusion

Given the current generation and matching rater and a realistic probability of finding a valid Bitcoin address, it could take around 121.8 years or more to find a single valid address (best case scenario). This highlights the impracticality of brute-forcing Bitcoin addresses to find ones with non-zero balances.

## License

The MIT License (MIT)

Copyright (c) 2024 Hunter Rose (x.com/HunterR0se)
