# Deepred-Mt

Given an RNA sequence consisting of a central cytidine flanked by 20
nucleotides on each side, Deepred-Mt scores if the central cytidine is
edited. This score is obtained from the motifs extracted by a deep
convolutional auto-encoder from the sequence. Deepred-Mt was
especially constructed to predict editing sites for angiosperm
mitochondrial RNA.

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
