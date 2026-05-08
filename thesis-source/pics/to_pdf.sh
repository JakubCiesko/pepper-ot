diagramname=$1
dot -Tpdf $diagramname.dot -o $diagramname.pdf && open $diagramname.pdf
