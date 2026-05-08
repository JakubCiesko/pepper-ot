diagramname=$1
dot -Tsvg $diagramname.dot -o $diagramname.svg && open $diagramname.svg
