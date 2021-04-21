# Deepred-Mt

In land plants, the editosome is a highly sophisticated molecular
machine able to bind organellar RNA molecules post-transcriptionally
to convert cytidines to uridines (C-to-U) at highly specific positions
called editing sites. RNA editing is governed by cis elements that
remain recalcitrant to characterization, limiting further advances on
editosome binding prediction. Deepred-Mt is a novel method to predict
editing editing for angiosperm mitochondrial RNA.

Given an RNA sequence consisting of a central cytidine flanked by 20
nucleotides on each side, Deepred-Mt scores if the central cytidine is
edited. This score is obtained from the motifs extracted by a deep
convolutional auto-encoder from the sequence.

Source code and instructions are provided for reproducibility of the
main results of "Deepred-Mt: Deep Representation Learning for
Predicting C-to-U RNA Editing in Plant Mitochondria," by A. A. Edera,
I. Small, D. H. Milone, M. V. Sanchez-Puerta (under review). Research
Institute for Signals, Systems and Computational Intelligence,
[sinc(i)](https://sinc.unl.edu.ar/).

## Datasets

* [train](https://foo.com)
* [Lophophytum mirabile](https://foo.com)

## Installation

```bash
pip install -U "deepredmt @ git+https://github.com/aedera/deepredmt.git"
```
