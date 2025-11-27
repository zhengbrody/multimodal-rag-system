# UV Package Manager Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python package and virtual environment management.

## Installation

### macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Homebrew (macOS)
```bash
brew install uv
```

### Windows
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Project Setup

### Initialize and Install Dependencies
```bash
# Create virtual environment and install all dependencies
uv sync

# Production dependencies only
uv sync --no-dev

# Frontend dependencies only (lightweight deployment)
uv sync --extra frontend
```

### Activate Virtual Environment
```bash
# Run commands with uv (recommended)
uv run python your_script.py

# Manual activation
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### Run Project
```bash
# Setup knowledge base
uv run python setup.py

# Start application
uv run python run.py
```

## Common Commands

### Add Package
```bash
uv add package-name              # Production dependency
uv add --dev package-name        # Development dependency
uv add package-name==1.2.3       # Specific version
```

### Remove Package
```bash
uv remove package-name
```

### Update Packages
```bash
uv sync --upgrade                # Update all packages
uv add --upgrade package-name    # Update specific package
```

### List Installed Packages
```bash
uv pip list
```

## Migration from pip

If you previously used pip + requirements.txt:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Remove old virtual environment (optional)
rm -rf venv/ env/ .venv/

# Initialize with uv
uv sync

# Run project
uv run python setup.py
uv run python run.py
```

## Troubleshooting

### Permission Issues
```bash
chmod +x ~/.cargo/bin/uv
```

### Python Version
Project requires Python >= 3.10. Check version:
```bash
uv run python --version
```

### Dependency Conflicts
```bash
uv cache clean
uv sync --reinstall
```

## Additional Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [UV Quick Start](https://docs.astral.sh/uv/)
