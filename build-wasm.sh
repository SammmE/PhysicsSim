#!/bin/bash
# Build script for WebAssembly target

echo "Building PhysicsSim for WebAssembly..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null
then
    echo "wasm-pack is not installed. Installing..."
    cargo install wasm-pack
fi

# Build the project
wasm-pack build --target web --out-dir pkg --release

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "To run the web version:"
    echo "1. Install a local web server (e.g., 'npm install -g http-server')"
    echo "2. Run: http-server . -p 8080"
    echo "3. Open http://localhost:8080 in your browser"
    echo ""
    echo "Note: Your browser must support WebGPU (Chrome 113+, Edge 113+)"
else
    echo "❌ Build failed!"
    exit 1
fi
