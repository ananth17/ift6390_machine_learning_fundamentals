projectStatement:
	pandoc --filter pandoc-citeproc project_statement.md --bibliography bibliography.bib -o projectStatement.pdf

# generate the pdf
project: biblio
	pdflatex project.tex
	evince project.pdf

# make the bibliography
biblio: project.bib
	latex project.tex
	bibtex project.aux
