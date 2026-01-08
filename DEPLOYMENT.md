# WebGPU Deployment Guide

This document provides instructions for deploying PhysicsSim to the web using WebGPU.

## Building for Web

### Prerequisites
- Rust toolchain with `wasm32-unknown-unknown` target
- wasm-pack installed (`cargo install wasm-pack`)

### Build Steps

1. **Build the WebAssembly package:**
   ```bash
   ./build-wasm.sh
   ```
   
   Or manually:
   ```bash
   wasm-pack build --target web --out-dir pkg --release
   ```

2. **Test locally:**
   ```bash
   # Using Python
   python3 -m http.server 8080
   
   # Or using Node.js
   npx http-server . -p 8080
   ```

3. **Open in browser:**
   Navigate to `http://localhost:8080/index.html`

## Browser Requirements

The web version requires a browser with WebGPU support:

- ✅ Chrome 113+ (Stable)
- ✅ Edge 113+ (Stable)
- ✅ Firefox Nightly (enable `dom.webgpu.enabled` in about:config)
- ✅ Safari Technology Preview (macOS)

## GitHub Pages Deployment

A GitHub Actions workflow (`.github/workflows/deploy.yml`) is included for automatic deployment to GitHub Pages.

### Setup

1. Go to your repository Settings → Pages
2. Set Source to "GitHub Actions"
3. Push to the `main` branch
4. The workflow will automatically build and deploy

Your site will be available at: `https://<username>.github.io/PhysicsSim/`

## Deployment to Other Platforms

### Netlify / Vercel

1. Build the WebAssembly package locally
2. Deploy the entire repository root (includes `index.html` and `pkg/` directory)
3. No special configuration needed - serve as static files

### Custom Server

Serve the following files from your web server:
- `index.html` (entry point)
- `pkg/PhysicsSim.js` (JavaScript bindings)
- `pkg/PhysicsSim_bg.wasm` (WebAssembly binary)

Ensure your server is configured to serve `.wasm` files with the correct MIME type:
```
application/wasm
```

## File Structure

```
PhysicsSim/
├── index.html              # Web application entry point
├── pkg/                    # WebAssembly build output
│   ├── PhysicsSim.js       # JavaScript bindings
│   ├── PhysicsSim_bg.wasm  # WebAssembly binary
│   └── ...
├── src/                    # Source code
├── Cargo.toml             # Rust configuration
└── build-wasm.sh          # Build script
```

## Troubleshooting

### "WebGPU is not supported in your browser"

Make sure you're using a browser with WebGPU support (see Browser Requirements above).

### "Failed to get WebGPU adapter"

Your GPU may not support WebGPU, or GPU drivers may need updating. Try:
1. Update your GPU drivers
2. Enable hardware acceleration in browser settings
3. Try a different browser

### Build Errors

If you encounter build errors:
1. Ensure you have the latest Rust toolchain: `rustup update`
2. Add the wasm target: `rustup target add wasm32-unknown-unknown`
3. Clean and rebuild: `cargo clean && ./build-wasm.sh`

## Features Available in Web Version

✅ Full physics simulation with particle system
✅ Interactive mouse controls (left/right drag)
✅ Real-time performance monitoring
✅ Dynamic graph plotting
✅ Live physics parameter adjustment
✅ Keyboard shortcuts (P, R, Tab, Space)

❌ File save/load (browser security restriction)

## Performance Notes

The web version uses WebGPU for GPU-accelerated rendering, providing performance comparable to the native desktop application. For best performance:

- Use Chrome/Edge with hardware acceleration enabled
- Close other GPU-intensive applications
- Use a modern GPU with WebGPU support
