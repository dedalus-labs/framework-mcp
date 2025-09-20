#!/bin/bash

echo "Setting up Dedalus Documentation MCP Server"
echo "============================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env.local ]; then
    echo "Creating .env.local file from template..."
    cp config/.env.example .env.local
    echo "Please edit .env.local to add your API keys (optional for local testing)"
fi

# Create docs directory if it doesn't exist
if [ ! -d docs ]; then
    echo "Creating docs directory..."
    mkdir -p docs
fi

echo ""
echo "Setup complete!"
echo ""
echo "To start the server:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the server: python src/server.py"
echo ""
echo "For Dedalus deployment:"
echo "  dedalus deploy ./src/server.py --name 'hackathon-docs-server'"
echo ""
echo "Add your documentation files to the 'docs/' directory"