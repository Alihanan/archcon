# ArchCon

ArchCon is currently a small Python package and local web application. It opens
in the browser, lets the user select or drag a text file from the filesystem,
and displays the file contents. The ArchCon model and thesis-specific processing
are not included yet.

## Requirements

- Python 3.10 or newer
- A modern web browser

## Install as a user

### From PyPI after the package is published

Create an isolated environment and install ArchCon:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install archcon
```

Start it from the terminal:

```bash
archcon
```

or from Python:

```python
from archcon import start

start()
```

The browser opens at `http://127.0.0.1:7860` by default. Stop the server with
`Ctrl+C`.

### From a wheel sent directly by the author

Install the wheel file:

```bash
python -m pip install archcon-0.1.0-py3-none-any.whl
```

Then run:

```bash
archcon
```

### Use the non-web Python API

```python
from archcon import read_text_file

status, content = read_text_file("example.txt")
print(status)
print(content)
```

## Command-line options

```bash
archcon --help
archcon --no-browser
archcon --port 8000
```

The default host is `127.0.0.1`, so the application is accessible only from the
same computer unless the host is explicitly changed.

## Development installation

Clone or extract the source project, enter its root directory, and run:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install --editable ".[dev]"
```

Run quality checks:

```bash
ruff format --check .
ruff check .
pytest
```

During development, start the application with:

```bash
archcon
```

Editable installation means changes under `src/archcon/` are used immediately.

# Initialize and upload to GitHub

Create an empty GitHub repository named `archcon`. Do not add a README, license,
or `.gitignore` on GitHub because this project already contains them. Then run
from the project root:

```bash
git init
git add .
git commit -m "Initial ArchCon package"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/archcon.git
git push -u origin main
```

If `origin` already exists, replace the remote URL instead:

```bash
git remote set-url origin https://github.com/YOUR_GITHUB_USERNAME/archcon.git
git push -u origin main
```

Alternatively, after installing and authenticating GitHub CLI:

```bash
gh auth login
gh repo create archcon --public --source=. --remote=origin --push
```

Use `--private` instead of `--public` if the repository should initially be
private. After the push, open the repository's **Actions** tab and confirm that
the `CI` workflow passes.

# Shipping a release

## 1. Update release metadata

Before the first public release, replace the placeholders in:

- `pyproject.toml`: author and GitHub URLs
- `LICENSE`: copyright holder

For every new release, change the version in `pyproject.toml`, for example:

```toml
version = "0.1.1"
```

A version already uploaded to PyPI cannot be replaced. Publish a new version for
any correction.

## 2. Build clean distributions

From the project root:

```bash
python -m pip install --editable ".[dev]"
ruff format --check .
ruff check .
pytest

rm -rf build dist                 # Linux/macOS
# Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue  # PowerShell

python -m build
python -m twine check dist/*
```

The build creates:

```text
dist/
├── archcon-0.1.0-py3-none-any.whl
└── archcon-0.1.0.tar.gz
```

## 3. Test the built wheel

Use a fresh virtual environment rather than testing only the editable source:

```bash
python -m venv wheel-test
source wheel-test/bin/activate    # Linux/macOS
# wheel-test\Scripts\Activate.ps1 # Windows PowerShell

python -m pip install dist/archcon-0.1.0-py3-none-any.whl
python -c "import archcon; print(archcon.__version__)"
archcon --help
```

You may send the `.whl` file directly to another user at this point.

## 4. Publish manually to TestPyPI first

Create a TestPyPI account, generate an API token, and upload:

```bash
python -m twine upload --repository testpypi dist/*
```

Test installation from TestPyPI while obtaining dependencies from normal PyPI:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  archcon
```

## 5. Publish manually to PyPI

Create a PyPI account and API token, then run:

```bash
python -m twine upload dist/*
```

Users can then install the release with:

```bash
python -m pip install archcon
```

## 6. Automated GitHub release pipeline

The repository includes:

```text
.github/workflows/ci.yml
.github/workflows/publish-pypi.yml
```

`ci.yml` runs formatting, linting, tests, and package building on pushes and pull
requests.

For automatic publication without storing a permanent PyPI password:

1. Push the project to `https://github.com/YOUR_GITHUB_USERNAME/archcon`.
2. On PyPI, configure a Trusted Publisher for that repository.
3. Set the workflow filename to `publish-pypi.yml`.
4. Set the GitHub environment name to `pypi`.
5. In GitHub repository settings, create an environment named `pypi`.
6. Create and publish a GitHub Release with a tag matching the package version,
   such as `v0.1.0`.

Publishing the GitHub Release starts the workflow, builds the wheel and source
archive, validates them, and uploads them to PyPI.

## Project structure

```text
archcon/
├── .github/workflows/
│   ├── ci.yml
│   └── publish-pypi.yml
├── src/archcon/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py
│   ├── cli.py
│   └── io.py
├── tests/
├── LICENSE
├── README.md
└── pyproject.toml
```

The reusable Python logic is kept separate from the Gradio interface. Future
model, training, evaluation, and visualization modules should remain usable
without importing the web interface directly.
