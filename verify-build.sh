#!/bin/bash
# Verification script to check if the project builds correctly for both native and web

set -e

echo "================================"
echo "PhysicsSim Build Verification"
echo "================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for Rust
echo "Checking for Rust installation..."
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Rust is not installed${NC}"
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi
echo -e "${GREEN}✓ Rust found: $(rustc --version)${NC}"
echo ""

# Check native build
echo "Checking native build..."
if cargo check --quiet 2>/dev/null; then
    echo -e "${GREEN}✓ Native build compiles successfully${NC}"
else
    echo -e "${RED}❌ Native build failed${NC}"
    exit 1
fi
echo ""

# Check for wasm target
echo "Checking for wasm32-unknown-unknown target..."
if rustup target list | grep -q "wasm32-unknown-unknown (installed)"; then
    echo -e "${GREEN}✓ WASM target is installed${NC}"
else
    echo -e "${YELLOW}⚠ WASM target not found. Installing...${NC}"
    rustup target add wasm32-unknown-unknown
    echo -e "${GREEN}✓ WASM target installed${NC}"
fi
echo ""

# Check WASM build
echo "Checking WASM build..."
if cargo check --target wasm32-unknown-unknown --quiet 2>/dev/null; then
    echo -e "${GREEN}✓ WASM build compiles successfully${NC}"
else
    echo -e "${RED}❌ WASM build failed${NC}"
    exit 1
fi
echo ""

# Check for wasm-pack
echo "Checking for wasm-pack..."
if command -v wasm-pack &> /dev/null; then
    echo -e "${GREEN}✓ wasm-pack found: $(wasm-pack --version)${NC}"
else
    echo -e "${YELLOW}⚠ wasm-pack not found${NC}"
    echo "  Install it with: cargo install wasm-pack"
fi
echo ""

# Summary
echo "================================"
echo -e "${GREEN}✓ All checks passed!${NC}"
echo "================================"
echo ""
echo "Next steps:"
echo "  • Build native:  cargo run --release"
echo "  • Build web:     ./build-wasm.sh"
echo "  • Test web:      python3 -m http.server 8080"
echo ""
echo "For more information, see:"
echo "  • README.md for general usage"
echo "  • DEPLOYMENT.md for web deployment details"
