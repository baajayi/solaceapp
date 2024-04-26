#!/bin/bash
# Activate virtual environment if necessary
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Gradio interface
python app.py

# Deploy to GitHub Pages
ghp-import -n -p -f .
