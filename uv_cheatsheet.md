# 📘 `uv` + Poetry Cheatsheet

`uv` is an ultra-fast Python package installer, often used as a drop-in replacement for `pip`, and can also be used **with Poetry projects** for significantly faster dependency resolution and installation.

---

## ✅ Installation

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
# OR
pipx install uv
```

---

## 🚀 Key Commands

| Task                      | Poetry Command              | `uv` Equivalent                     |
|---------------------------|-----------------------------|--------------------------------------|
| Install dependencies      | `poetry install`            | `uv pip install -r requirements.txt` |
| Add dependency            | `poetry add requests`       | `uv pip install requests`           |
| Remove dependency         | `poetry remove requests`    | `uv pip uninstall requests`         |
| Update all packages       | `poetry update`             | `uv pip install -U -r requirements.txt` |
| Lock dependencies         | `poetry lock`               | `uv pip compile` (w/ `pip-tools`)   |
| Sync w/ lockfile          | `poetry install --sync`     | `uv pip sync` (w/ `pip-tools`)      |

---

## 🧩 Using `uv` in a Poetry Project (Drop-in Style)

You can keep using `poetry` for dependency management and use `uv` just for fast installs.

### Step 1: Export requirements

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

### Step 2: Use `uv` to install

```bash
uv pip install -r requirements.txt
```

Or for dev dependencies:

```bash
poetry export -f requirements.txt --with dev --output requirements-dev.txt --without-hashes
uv pip install -r requirements-dev.txt
```

---

## 📦 Combining with `venv`

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
---

## ✅ Step-by-Step: Using uv inside Poetry's virtual environment
Activate the Poetry environment:
```bash
poetry shell
```
- This activates the virtual environment created and managed by Poetry.
- Install dependencies using uv: Now that you're inside the environment, you can safely run:

```bash
uv pip install -r requirements.txt
```
- This will install dependencies using uv's ultrafast resolver into the same virtual environment Poetry created.

---

## 🔄 Sync Script (Optional)

You can automate syncing your lockfile to your env using a shell script:

```bash
poetry export -f requirements.txt --without-hashes > requirements.txt
uv pip install -r requirements.txt
```

---

## 🔧 Tips

- `uv` is blazing fast ⚡ due to Rust implementation.
- Use `uv` for installs even inside CI for better performance.
- Great for Docker images where speed matters.

---
## 🧩 Bonus Tip: Find Poetry's venv path without activating it
- If you want to use uv programmatically or from another script:
```bash
poetry env info --path
```
- Then you can activate or use that environment path explicitly, e.g.,
```bash
source $(poetry env info --path)/bin/activate
uv pip install -r requirements.txt
```
---

## 📚 Resources

- Official: https://github.com/astral-sh/uv
- Poetry: https://python-poetry.org/
- Speed comparison: https://github.com/astral-sh/uv#benchmarks

