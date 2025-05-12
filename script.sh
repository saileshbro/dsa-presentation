#!/bin/bash
set -e

# Step 1: Generate Quicksort visualization TeX
python3 programs/quicksort/quicksort_texgen.py

# Step 2: Compile the presentation (run twice for references)
cd presentation
pdflatex -interaction=nonstopmode presentation.tex
biber presentation || true
pdflatex -interaction=nonstopmode presentation.tex
pdflatex -interaction=nonstopmode presentation.tex

cd ..
echo "Build complete. PDF is at presentation/presentation.pdf"