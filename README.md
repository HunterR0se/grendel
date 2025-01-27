Grendel

**Author:** [Hunter Rose](https://x.com/HunterR0se)

## Overview

Grendel is a tool designed to generate and test cryptographic keys against known Bitcoin addresses. The project supports both CPU and NVIDIA GPU acceleration for key generation.

Hat tip to;
https://github.com/Harold-Glitch/bitreverse/blob/master/secp256k1.cuh
https://github.com/XopMC/CudaBrainSecp/tree/main/GPU

Key features include:

- Detection of CPU or NVIDIA GPU availability on startup.
- Integration with LevelDB for storing known addresses.
- Periodic loading of new Bitcoin blocks during runtime.
- Logging of found addresses with their seeds, private keys, and balances.

## Getting Started

If you want to short-circuit all the compilation and everything else, here's the latest binary for Linux. It will not (and should not) run on Windoze.

### Download Binary

1. **Download the Pre-compiled Binary:**

    ```bash
    wget [BINARY_LINK_HERE] -O grendel
    ```

2. **Make it Executable and Move to System Path:**

    ```bash
    chmod +x grendel
    sudo mv grendel /usr/local/bin/
    ```

3. **Verify Installation:**

    ```bash
    grendel
    ```

4. **Ensure Address Files:**
    - Make sure your Bitcoin address files are in the `~/.bitcoin` directory
    - If not present, run Bitcoin Core first to sync the blockchain

> **Note:** The binary is compiled for x86_64 Linux systems. For other architectures, please build from source.

### Typical Loading Screen

[🎮 -GPU-] ───────────────────────────────────────────────────────────
[🎮 -GPU-] CPU vs GPU Benchmark (5,000,000 addresses)
[🎮 -GPU-] ───────────────────────────────────────────────────────────
[🎮 -GPU-] CPU: 16.834972167s (297,000 keys/sec)
[🎮 -GPU-] GPU: 651.62487ms (7,673,126 keys/sec)
[🎮 -GPU-] Speedup: 25.84x
[🎮 -GPU-] ───────────────────────────────────────────────────────────
[🔍 -INFO] Loading Addresses: 2.34GB
[🔍 -INFO] Total Addresses: 55,006,296 (67.6 seconds)
[📁 -DATA] ───────────────────────────────────────────────────────────
[📁 -DATA] Loading addresses into memory...
[🧠 -MEM-] Loaded 55,006,296 addresses in 35.195 seconds

### Prerequisites

- A Linux-based operating system (e.g., Debian)
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

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with your changes.

### Real-World Consideration

The actual probability of randomly generating a key that matches an existing address with a non-zero balance is extremely low, perhaps on the order of 1 in 10^12 or even lower, due to the vast number of possible keys and the relatively small number of addresses with non-zero balances.

#### Conclusion

Given the rates you've provided and a realistic probability of finding a valid Bitcoin address, it could take around 121.8 years or more to find a single valid address (best case scenario). This highlights the impracticality of brute-forcing Bitcoin addresses to find ones with non-zero balances.

## License

The MIT License (MIT)

Copyright (c) 2024 Hunter Rose (x.com/HunterR0se)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
