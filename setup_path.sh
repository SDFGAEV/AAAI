#!/bin/bash
# Add C-ACT project paths to Python environment
# Run once per venv: bash setup_path.sh
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE/cact.pth" <<EOF
$(pwd)
$(pwd)/src
$(pwd)/minerl
EOF
echo "Done. Restart your Python session or source venv/bin/activate again."
