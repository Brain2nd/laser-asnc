# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

NeurIPS 2026 submission built on the official NeurIPS 2026 LaTeX template. The live paper is "LASER: A High-Fidelity Spike Representation SNN Framework With Surrogate-Free Training" (Zhengzheng Tang). `neurips_2026.sty` and `neurips_2026.tex` are the unmodified template files shipped by the conference — do **not** edit them unless explicitly instructed; they may need to be reset to upstream for submission.

## Build

Standard natbib-bibliography cycle (run from the repo root):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`latexmk -pdf main.tex` does the same in one command. Replace `main` with `main_allen`, `main_allen_v2`, `main_allen_v3`, `main_allen_v4`, or `neurips_2026` to build a different draft. `main.pdf` is the currently checked-in compiled artifact.

## Track selection

The track is chosen by the `\usepackage[...]{neurips_2026}` option at the top of the document (see `neurips_2026.tex:10-45` for the full list):

- no option / `main` — Main Track (double-blind, default)
- `position`, `eandd`, `creativeai` — other tracks (`eandd` supports `nonanonymous`)
- `sglblindworkshop`, `dblblindworkshop` — workshop tracks; **both** require `\title{}` and `\workshoptitle{}`
- `preprint` — arXiv-style non-anonymous preprint
- Add `final` after the track option for the camera-ready (e.g. `[main, final]`)
- `nonatbib` disables natbib if it clashes with another package

Switching the option changes anonymization, footer track name, and line numbers automatically via `neurips_2026.sty`.

## Source layout

- `main.tex` — primary working draft of the LASER paper. Sections: Introduction, Related Works, Methodology, Experiments, Conclusion, then appendices (Entropy Preservation of BSE, ASNC Error Bound, ASNC Formal Specification). Bibliography uses `\bibliographystyle{plainnat}` with `references.bib`.
- `main_allen.tex`, `main_allen_v{2,3,4}.tex` — parallel author drafts with a different structure (Preliminaries section, expanded appendix including "Transformer-Specific Operations", "End-to-End Error Bound", "STE Gradient Analysis"). `v4` is the most recent / longest (~2200 lines).
- `neurips_2026.tex` — upstream template with usage examples; keep as reference.
- `checklist.tex` — the NeurIPS reproducibility checklist. It must appear **after** the bibliography and appendix and is included via `\input{checklist.tex}` (see `neurips_2026.tex:490`). The working `main.tex` currently inlines the checklist body directly rather than using `\input`. Papers without the checklist are desk-rejected, and the instruction block at the top of `checklist.tex` must be deleted before submission while keeping the section heading and all questions.
- `references.bib` — shared bibliography for all drafts.
- `figures/` — PDF figures (`error_diffusion.pdf`, `error_diffusion1.pdf`). Reference with `\includegraphics{figures/...}`.

## Editing conventions specific to this project

- The LASER drafts define a Tol Bright colorblind-safe palette (`TolBlue`, `TolCyan`, `TolGreen`, `TolYellow`, `TolRed`, `TolPurple`, `TolGrey`) near the top of `main.tex`. Reuse these names rather than introducing new color definitions.
- Theorem environments (`theorem`, `proposition`, `lemma`, `corollary`, `definition`, `assumption`, `remark`) are numbered together within each section — keep that shared counter when adding new environments.
- `cleveref` is loaded with `capitalize,noabbrev` **after** `hyperref`; preserve that order when adding packages.
- Many introduction paragraphs are commented-out Chinese drafts followed by English rewrites. Leave them in place unless the task is a cleanup pass — they are the author's working notes.
- For the anonymous submission, do not add identifying info to `\author{}`; the `final` option handles de-anonymization. `eandd` is the only track with an opt-in `nonanonymous` flag.

## Checklist answers

Answers in `checklist.tex` use the macros `\answerYes{}`, `\answerNo{}`, `\answerNA{}`, `\answerTODO{}` and `\justificationTODO{}`. Every question needs a 1–2 sentence justification, including `\answerNA{}`. Do not modify question text or guideline bullets — only replace the answer and justification macros.
