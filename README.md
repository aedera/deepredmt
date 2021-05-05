# Deepred-Mt

In land plants, the editosome is a highly sophisticated molecular machine able
to bind organellar RNA molecules post-transcriptionally. It converts cytidines
to uridines (C-to-U) at highly specific RNA positions called editing
sites. RNA editing is governed by cis elements that remain recalcitrant to
characterization, limiting further advances on genetic engineering techniques
for RNA manipulation.

Deepred-Mt is a novel method to predict C-to-U editing sites in angiosperm
mitochondrial RNA. Given an RNA sequence consisting of a central cytidine
flanked by 20 nucleotides on each side, Deepred-Mt scores if the central
cytidine is edited. This score is computed using motifs extracted by a
convolutional approach from the sequence.

Source code and instructions are provided for reproducibility of the main
results of "Deepred-Mt: Deep Representation Learning for Predicting C-to-U RNA
Editing in Plant Mitochondria," by A. A. Edera, I. Small, D. H. Milone, and
M. V. Sanchez-Puerta (under review). Research Institute for Signals, Systems
and Computational Intelligence, [sinc(i)](https://sinc.unl.edu.ar/).

<figure>
  <p align="center">
  <img src=fig/model-architecture.png alt="Deepred-mt" width="940" style="vertical-align:middle"/>
  </p>

  <figcaption>Architecture of Deepred-Mt. </figcaption>
</figure>


## Installation

```bash
pip install -U "deepredmt @ git+https://github.com/aedera/deepredmt.git"
```

## Example usage

### Command line

After installing, you can use Deepred-Mt on the command line to predict
C-to-U editing sites for a desired fasta file:

```bash
deepredmt data/lopho.fas
```

This searches the fasta file for all the cytidines and then predicts if they
are edited based on their surrounding nucleotides.

### Notebooks

* [Notebook 1](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/01_prediction_from_fasta.ipynb)
  shows how to use Deepred-Mt on the command line to predict C-to-U editing
  sites in a fasta file.

*
  [Notebook 2](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/02_reproduce_comparative_analysis.ipynb)
  reproduces results of the manuscript in which the predictive prediction of
  Deepred-Mt is compared with that of a state-of-the art method for predicting
  editing sites.

* [Notebook 3](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/03_deepredmt_training.ipynb)
  shows how to train Deepred-Mt from scratch.

## Datasets

* [Training data](./data/training-data.tsv.gz). It contains 41-bp
  nucleotide windows whose center positions are either unedited (C) or
  edited (E) cytidines. Nucleotide windows are labeled according to
  both the nucleotide in their central positions (0/C, 1/E) and
  editing extents.

* [Task-related sequences](./data/task-related-sequences.tsv.gz). This
  dataset was constructed by using the task-related augmentation
  strategy. It contains 41-bp nucleotide windows whose center
  positions are thymidines homologous to the edited sites in the
  training data.

* [Control data](./data/control-data.tsv.gz). This dataset was constructed
  from the training dataset by injecting the fake editing signal "GGCG" in the
  downstream regions of the nucleotide windows labeled as 1. Each time this
  fake signal was injected, one of its four nucleotides was mutated randomly
  with certain probability.

* [_Lophophytum mirabile_ data](./data/lopho-data.tsv.gz). It contains
  41-bp nucleotide windows obtained from the mitochondrial
  protein-coding sequences of a flowering plant called _Lophophytum
  mirabile_. The C-to-U editing sites of these sequences were
  experimentally identified in a [previous
  study](https://doi.org/10.1111/nph.16926) by using deep RNA
  sequencing.

Look at this [README](./data) file for more information on the format of these
datasets.
