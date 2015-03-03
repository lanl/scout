pdflatex:= $(shell which latexmk)

all: reference

reference:
	$(pdflatex) --pdf reference

clean:
	$(pdflatex) --c reference
	-@/bin/rm -f reference.bbl
	-@/usr/bin/find . -name '*~' -exec rm -f {} \;

distclean: clean
	$(pdflatex) --C  reference

