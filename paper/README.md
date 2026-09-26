# Manuscript

`main.tex` is the IEEEtran journal manuscript. Its bibliography is split between `references.bib` and `references_added.bib`. The compiled version is `main.pdf`; all figures used by the manuscript are in `figures/`.

Build with a TeX distribution providing IEEEtran, newtx, latexmk, and BibTeX:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Figure 11 reports controlled architecture-model comparisons. The supplied simulator reproduces those five configurations; see [the reproduction guide](../docs/reproduction.md).

This repository version includes the authors, affiliations, funding acknowledgment, corresponding-author information, and a link to the public simulator in the abstract.

The archived plotting scripts and inputs for Figures 11 and 12 are in
`figure_sources/`. Install `python -m pip install -e ".[figures]"` from the
repository root to use them. They write exports beside each script; they do
not overwrite the PDFs used by the manuscript. Arial should be installed to
match the paper's typography.
