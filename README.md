# Deepred-Mt

In land plants, the editosome is a highly sophisticated molecular machine able
to bind organellar RNA molecules post-transcriptionally. It converts cytidines
to uridines (C-to-U) at highly specific RNA positions called editing
sites. RNA editing is governed by cis elements that remain recalcitrant to
characterization, limiting further advances on genetic engineering techniques
for RNA manipulation.

Deepred-Mt is a novel method to predict editing editing for angiosperm
mitochondrial RNA. Given an RNA sequence consisting of a central cytidine
flanked by 20 nucleotides on each side, Deepred-Mt scores if the central
cytidine is edited. This score is computed using motifs extracted from the
sequence with a convolutional arquitecture.

Source code and instructions are provided for reproducibility of the
main results of "Deepred-Mt: Deep Representation Learning for
Predicting C-to-U RNA Editing in Plant Mitochondria," by A. A. Edera,
I. Small, D. H. Milone, M. V. Sanchez-Puerta (under review). Research
Institute for Signals, Systems and Computational Intelligence,
[sinc(i)](https://sinc.unl.edu.ar/).

<figure>
  <p align="center">
  <img src=fig/model-architecture.png alt="Deepred-mt" width="940" style="vertical-align:middle"/>
  </p>

  <figcaption>Architecture of Deepred-mt.</figcaption>
</figure>


## Datasets

* [train](https://foo.com)
* [Lophophytum mirabile](https://foo.com)

## Installation

```bash
pip install -U "deepredmt @ git+https://github.com/aedera/deepredmt.git"
```
