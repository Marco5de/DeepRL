all: main.pdf
	$(info ARTIFACTS:main.pdf)

.PHONY: main.pdf
main.pdf: main.tex
	latexmk  -pdf  main.tex

.PHONY: clean
clean:
	latexmk  -c
	rm main.tdo main.bbl main.run.xml main-blx.bib