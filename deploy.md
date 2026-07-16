# pixi based deployment

Rapida provides a suite of tools for real-time crisis assessment, relying on a vast ecosystem of software libraries. 
Because many of these dependencies are written in C/C++ for performance and require complex, dynamic linking to 
system-level libraries, installation can be notoriously difficult. 
To eliminate this barrier, Rapida leverages pixi — a **fast, modern, and highly reproducible** package manager 
that guarantees a seamless setup across all platforms.

Pixi defaults to the biggest Conda package repository, conda-forge, which contains over 30,000 packages.
## Install pixi

Refer to [pixi installation](https://pixi.prefix.dev/latest/#installation) docs

## Download

**Linux & macOS**
```bash
curl -o [https://pixi.sh/install.sh](https://pixi.sh/install.sh) | sh
