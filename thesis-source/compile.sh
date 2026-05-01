awk '/^@/ {if (seen[$0]++) skip=1; else skip=0} !skip' bibliography.bib > bibliography_clean.bib
latexmk -C -jobname=thesis fi-pdflatex.tex && latexmk -pdf -jobname=thesis fi-pdflatex.tex
