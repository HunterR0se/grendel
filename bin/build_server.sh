#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Add timestamp function
format_timestamp() {
    date "+%B %d, %Y at %I:%M%p %Z"
}

# Clear screen and add blank lines
clear
echo
echo

echo -e "${BLUE}🚀 Starting Server Setup $(format_timestamp)...${NC}"

# Update System
echo -e "${BLUE}🔄 Updating system...${NC}"
if ! sudo apt update; then
    echo -e "${RED}❌ System update failed${NC}"
    exit 1
fi

# Install Basic Tools
echo -e "${BLUE}🔧 Installing basic tools...${NC}"
if ! sudo apt install -y git vim zsh tmux; then
    echo -e "${RED}❌ Installation of basic tools failed${NC}"
    exit 1
fi

# Configure Zsh as Default Shell
echo -e "${BLUE}🔄 Configuring Zsh as default shell...${NC}"
if ! chsh -s $(which zsh); then
    echo -e "${RED}❌ Failed to configure Zsh as default shell${NC}"
    exit 1
fi

# Install Oh-My-Zsh
echo -e "${BLUE}🔧 Installing Oh-My-Zsh...${NC}"
if ! RUNZSH=no sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"; then
    echo -e "${RED}❌ Installation of Oh-My-Zsh failed${NC}"
    exit 1
fi

# Set Time Zone
echo -e "${BLUE}🌍 Setting time zone...${NC}"
if ! sudo ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime; then
    echo -e "${RED}❌ Failed to set time zone${NC}"
    exit 1
fi

# Install Dependencies for Bitcoin Core
echo -e "${BLUE}🔧 Installing dependencies for Bitcoin Core...${NC}"
if ! sudo apt install -y apt-transport-https curl gnupg wget tmux htop; then
    echo -e "${RED}❌ Installation of dependencies failed${NC}"
    exit 1
fi

# Download Bitcoin Core
echo -e "${BLUE}⬇️  Downloading Bitcoin Core...${NC}"
mkdir -p install
cd install
if ! wget https://bitcoincore.org/bin/bitcoin-core-28.0/bitcoin-28.0-x86_64-linux-gnu.tar.gz; then
    echo -e "${RED}❌ Download of Bitcoin Core failed${NC}"
    exit 1
fi

# Extract and Install Bitcoin Core
echo -e "${BLUE}📦 Extracting and installing Bitcoin Core...${NC}"
if ! tar zxvf bitcoin-28.0-x86_64-linux-gnu.tar.gz; then
    echo -e "${RED}❌ Extraction of Bitcoin Core failed${NC}"
    exit 1
fi
if ! sudo install -m 0755 -o root -g root -t /usr/local/bin bitcoin-28.0/bin/*; then
    echo -e "${RED}❌ Installation of Bitcoin Core failed${NC}"
    exit 1
fi

# Cleanup
echo -e "${BLUE}🧹 Cleaning up...${NC}"
cd ~
rm -rf install

echo -e "${GREEN}✅ Server setup complete!${NC}"

# Print final status
echo -e "\n${BLUE}📋 Summary:${NC}"
echo -e "  🕒 $(format_timestamp)"
echo -e "  🎉 Server setup completed successfully!"
echo -e "  ⌛ $(format_timestamp)"

# Configure Bitcoin
echo -e "${BLUE}🔧 Configuring Bitcoin...${NC}"
echo -e "${YELLOW}Run this command on the server:${NC}"
echo -e "${BLUE}mkdir -p ~/.bitcoin${NC}"
echo -e "${YELLOW}Run this command on the localhost:${NC}"
echo -e "${RED}scp bitcoin/bitcoin.conf SSH_HOST:~/.bitcoin/bitcoin.conf${NC}"
