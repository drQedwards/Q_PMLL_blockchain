
# 🐍 **my-lib** — dog‑fooding the `pypm` prototype

[![CI](https://github.com/your-org/my-lib/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/my-lib/actions)
[![PyPI version](https://img.shields.io/pypi/v/my-lib.svg)](https://pypi.org/project/my-lib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A tiny Python package whose **sole job is to stress-test  
> `pypm` — the minimal C-based package‑manager prototype.**  
> Think of *my-lib* as the canary in the dependency‑graph coal mine:  
> if it installs, runs, and uninstalls cleanly, `pypm` is healthy. 🪺🛠️

---

## ✨ Why does this exist?

During the 2025 rewrite of `pypm` (now just **400 LOC of C** 🎉), we needed a *real* project that:

1. **Exercised every pypm code path**  
   *PEP 621 metadata → lock‑file resolution → venv build → plugin hooks → sandboxed execution*—the works.

2. **Pulled in both lightweight and heavyweight wheels**  
   From *requests* (65 kB) to **PyTorch 2.3.1** (≈ 780 MB), proving the wheel cache, hash verifier, and progress bars all behave.

3. **Remained 100 % deterministic**  
   Every wheel, hash, and transitive dep is frozen in `pypm.lock`. Re‑sync on any machine → identical bits.

---

## 🔧 Key features (of the **project**, not the PM)

| Area | What we test | Why it matters |
|------|--------------|----------------|
| **Runtime graph** | `requests`, `urllib3`, `charset‑normalizer`, etc. | Classic “small wheels” used in pretty much every app. |
| **Heavy wheel** | `torch 2.3.1` CPU‑only | Validates multi‑hundred‑MB downloads, resumable caching, and SHA‑256 verification. |
| **Conditional deps** | `tomli` only on < Python 3.11 | Confirms marker parsing and selective installs. |
| **Dev extras** | `black`, `pytest`, `mypy` | Makes sure *extras* land in a **separate group** so prod images stay slim. |
| **Plugin system** | `auditwheel` & `s3cache` stubs | pypm dlopen()s `pypm_<name>.so` and passes a rich context struct. |

---

## 🚀 Quick‑start

```bash
# Clone the repo
git clone https://github.com/your-org/my-lib
cd my-lib

# Rebuild the deterministic venv
pypm sync               # reads pypm.lock, verifies hashes, installs wheels

# Run the tiny demo module (prints installed torch version, etc.)
pypm run python -m my_lib.demo
```

Need a REPL inside the sandboxed venv? `pypm shell`

---

## 📂 Project layout

```
my-lib/
├─ pyproject.toml   # canonical metadata + tool.pypm table
├─ pypm.lock        # exact dependency graph w/ hashes
├─ src/my_lib/
│  ├─ __init__.py
│  └─ demo.py       # one‑liner: print("torch", torch.__version__)
└─ .venv/           # auto‑managed; hidden from git
```

---

## 🛠️ Developing

```bash
# Keep lock fresh after editing deps
pypm lock          # resolves & rewrites pypm.lock

# Run tests + type‑check
pypm run pytest
pypm run mypy src/
```

---

# 📝 Release Notes — **my-lib v0.1.0** & **pypm v0.1‑alpha**  
*2025‑06‑25  |  First public cut*  

## 🚀 Highlights
| Piece | Why you should care |
|-------|--------------------|
| **pypm in 400 LOC of C** | A proof‑of‑concept, single‑binary package manager that resolves, locks, verifies, builds a sandboxed venv, and drops you into it. |
| **Deterministic dependency graph** | `pypm.lock` pins every wheel (SHA‑256 + size) from *requests* ➜ *PyTorch 2.3.1* (≈ 780 MB). Re‑sync on any box → identical bits. |
| **Plugin ABI (dlopen)** | Add behaviours at each lifecycle stage (`pre‑sync`, `post‑sync`, etc.) with a `pypm_<name>.so`; shipped stubs: `auditwheel`, `s3cache`. |
| **my-lib demo code** | Minimal `src/my_lib/demo.py` that prints your live torch/runtime stack so you can sanity‑check the build. |
| **CI‑ready** | Designed so GitHub Actions can call `pypm sync && pypm run pytest` and reproduce your local environment byte‑for‑byte. |

## ✨ What’s new in **pypm 0.1‑alpha**

* **Commands** – `init`, `lock`, `sync`, `run`, `shell` (+ `--verbose`).  
* **Integrated venv builder** – auto‑creates `.venv/` when absent; deletes on Ctrl‑C.  
* **Lock resolver stub** – reads/writes TOML with a content‑hash header (full resolver lands in 0.2).  
* **Wheel cache** – downloads into `~/.cache/pypm/wheels`, resumes partial files, verifies SHA‑256.  
* **Cross‑platform skeleton** – tested on Linux; macOS & Windows shims left as TODO comments.  
* **Graceful plugin failure** – non‑zero return from `pypm_hook()` aborts the run with a friendly message.  
* **Signal handling** – Ctrl‑C / SIGINT cleans up mkdtemp sandbox before exit.

## 📦 What’s inside **my-lib 0.1.0**

* **Core deps** – `requests 2.32.3`, `torch 2.3.1`, plus transitive stack (`filelock`, `networkx`, `sympy`, `typing‑extensions`).  
* **Dev extras** – `black 24.4`, `pytest 8.0`, `mypy 1.10` to exercise the *optional‑dependencies* and *extras* paths.  
* **Project metadata** – pure PEP 621 in `pyproject.toml`; all pypm knobs folded under `[tool.pypm]`.  
* **Extended README** – badges, quick‑start, rationale, project layout, and development workflow.

## 🛠 Breaking / Gotchas

* **Experimental lock format 1.0** – will change in pypm 0.2; regen with `pypm lock` after upgrading.  
* **CPU‑only torch wheel** – swap in a CUDA/ROCm wheel manually if you need GPU; remember to pin a new hash!  
* **Sympy & typing‑extensions hashes** – placeholder `sha256:TODO‑sha256` lines until you run `pypm lock --refresh` (PyPI JS obfuscation workaround).  
* **Linux‑centric paths** – Windows users must adjust `.so` ➜ `.dll` in plugin loader & tweak path separators.

## 🗺 Roadmap

| Version | Planned goodies |
|---------|-----------------|
| **0.2** | Real parallel resolver, update‑check command, per‑wheel Merkle tree for tamper‑proofing. |
| **0.3** | Namespace sandbox (`pivot_root` + user namespaces) for near‑Hermetic builds without root. |
| **0.4** | Mac & Windows support; signed lock files; interactive progress UI with rich‑style bars. |
| **1.0** | Stabilised lock spec, plugin API freeze, binary releases via GH Releases. |

## 🔄 Upgrade instructions

```bash
# Inside an existing clone
git pull origin main            # grab these release notes
pypm lock --refresh             # fills any TODO‑sha256 wheels
pypm sync                       # idempotent env rebuild
pypm run python -m my_lib.demo  # confirm torch prints 2.3.1
```

## 🙏 Acknowledgements

Huge thanks to everyone who stress‑tested the alpha (especially **Dr. Josef Kurk Edwards** & **John Trompeter**—healing vibes post‑MRSA!).  
Bug reports → Issues tab; patches → PRs welcome. Let’s make deterministic Python builds boringly reliable 💖.

*(c) 2025 MIT License — “If you can’t reproduce it, you don’t really own it.”*
