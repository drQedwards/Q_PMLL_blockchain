# PPM by Dr. Q Josef Kurk Edwards &qchains
The Python Package Manager (# **pypm** – the “npm-style” package manager for Python  
*C-powered core · reproducible installs · plugin-friendly · workspace-aware*

![CI](https://img.shields.io/badge/build-passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![version](https://img.shields.io/badge/pypm-0.0.2-yellow)

> **TL;DR**: `pypm` aims to be a **single command** that handles everything from creating a
> virtual-env to publishing wheels—fast, deterministic, and hackable.  
> The current proof-of-concept is ~500 LOC of portable C that already
> boots a shell, diagnoses broken build chains, runs dynamically-loaded plugins,
> and produces hermetic bundles for air-gapped deploys.

---

## ✨ Features (0.0.2)

| Command                     | What it does                                                               |
|-----------------------------|---------------------------------------------------------------------------|
| `pypm doctor`               | Checks for Python headers, a C compiler, OpenSSL, WASI toolchain, …       |
| `pypm sandbox [-d DIR]`     | Drops you into an ephemeral temp dir (or custom DIR) with a full shell    |
| `pypm plugin add NAME SRC`  | Downloads a `.so` plugin (from URL or path) into `~/.pypm/plugins/`       |
| `pypm plugin run NAME …`    | Executes `pypm_plugin_main()` inside the named plugin                     |
| `pypm pypylock [-o FILE]`   | Bundles **every wheel + interpreter** into `dist/venv.tar.gz` (or FILE)   |
| `pypm version`              | Prints the current CLI version                                            |

*Road-mapped:* SAT dependency solver, parallel wheel cache, workspaces with
single lockfile, WASM wheel resolution, Conda & Poetry import plugins.

---

## 🔧 Building from source

```bash
# System deps: a C11 compiler, libcurl, libdl (both standard on Linux/macOS),
# and tar/libarchive if you want pypylock bundles.

git clone https://github.com/yourname/pypm.git
cd pypm
cc -Wall -Wextra -ldl -lcurl -o pypm pypm.c
./pypm doctor

# Diagnose your dev box
./pypm doctor

# Spin up a throw-away REPL playground
./pypm sandbox
# ...hack around, then exit – the temp dir vanishes.

# Add the Conda bridge plugin
./pypm plugin add conda https://example.com/plugins/conda.so

# Use it
./pypm plugin run conda install numpy==1.28.2

# Ship an offline bundle
./pypm pypylock -o /tmp/my-app.tgz

pypm.c                ← single-file CLI core (will split into modules)
include/              ← platform shims, TOML/YAML parsers (planned)
plugins/              ← sample plugins (conda bridge, poetry-import, hello-world)
docs/                 ← design notes, C API for plugin authors
README.md             ← you are here

// hello.c
#include <stdio.h>
int pypm_plugin_main(int argc, char **argv) {
    puts("Hello from a plugin 👋");
    return 0;
}

cc -shared -fPIC -o hello.so hello.c
mkdir -p ~/.pypm/plugins
mv hello.so ~/.pypm/plugins/
pypm plugin run hello

## 📚 Extended Description

### 1. Why another Python package manager?

Python’s packaging ecosystem is a vibrant—but fragmented—bazaar of tools:  
`pip` for installing, `venv` or `virtualenv` for isolating, `pipx` for app-style installs,  
`poetry`/`hatch`/`pdm` for workflow sugar, and Conda/Mamba for binary heft.  
Each excels at one slice yet leaves sharp edges when you stitch them together.

`pypm` is an **opinionated reboot** that cherry-picks the best ideas from npm, Cargo, and
Rust’s `uv`, then bakes them into a single, ultra-portable binary:

* **One command** (`pypm`) drives the _entire_ lifecycle.
* Determinism by default—every build is bit-for-bit reproducible.
* A C core keeps startup under ~15 ms and has zero runtime deps aside
  from `libc`, `libdl`, and `libcurl`.
* A **first-class plugin ABI** lets you graft in Conda, Poetry import, or even
  _your own_ solver written in Rust, Go, or Zig.

### 2. Guiding principles

| Principle | Manifestation in `pypm` |
|-----------|-------------------------|
| **Deterministic everywhere** | Lockfile pins version _and_ SHA-256 + optional Sigstore signature.  The resolver prefers “least-churn” upgrades so CI diffs stay legible. |
| **Speed trumps completeness** | Parallel wheel downloads, a content-addressed global cache, and a lazy SAT solver that stops at the first minimal solution. |
| **Extensibility beats bloat** | Core CLI is ~500 LOC; everything else (Conda, WASI, Poetry import, Docker image builds) lives in plugins. |
| **Cross-platform parity** | Workspace logic, tar bundling, and plugin loading all wrap POSIX + Win32 in thin shims—no “Linux-first” shortcuts. |
| **Security is not an add-on** | `pypm audit` talks to OSV & CVE feeds; lockfile embeds supply-chain metadata; `doctor` surfaces missing SSL/PGP bits _before_ you install. |

### 3. Architectural overview

```text
┌───────────────┐
│ pypm (CLI)    │  ←─ Typer-like command parser in C
└───────┬───────┘
        │
        ▼
┌───────────────┐     ┌─────────────┐     ┌──────────────┐
│ Workspace     │◀───▶│ Resolver    │◀───▶│ Wheel Cache  │
│ (TOML/YAML)   │     │ (SAT + PEP) │     │ (~/.cache)   │
└───────────────┘     └─────┬───────┘     └─────┬────────┘
                            │                  │
                            ▼                  ▼
                       ┌──────────┐      ┌────────────┐
                       │ Env Mgr   │      │ Plugin Host│
                       │ (.venv)   │      │ (dlopen)   │
                       └──────────┘      └────────────┘
