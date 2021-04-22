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

  <figcaption>Architecture of Deepred-Mt.</figcaption>
</figure>


## Installation

```bash
pip install -U "deepredmt @ git+https://github.com/aedera/deepredmt.git"
```

## Example usage

### Command line

After installing, you can execute Deepred-Mt on the command line to predict
editing sites from a desired fasta file:

```bash
deepredmt fasta/lopho.fas
```

This searches the fasta file for all the cytidines and then predicts if they
are editing sites based on their surrounding nucleotides.

### Notebooks

* [Notebook 1](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/01_prediction_from_fasta.ipynb)
  shows how to use Deepred-Mt on the command line to predict C-to-U editing
  sites in a fasta file.

*
  [Notebook 2](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/02_reproduce_comparative_analysis.ipynb)
  reproduces the results obtained when comparing the predictive prediction of
  Deepred-Mt an [PREP-Mt](http://prep.unl.edu/) on the protein-coding
  sequences of 21 mitochondrial genomes of diverse angiosperms.

## Datasets

* [Training data](./datasets/training-data.tsv.gz). These data were used to
  train Deepred-mt, and evaluate its predictive performance. It contains 41-bp
  nucleotide windows whose center positions are either cytidines or C-to-U
  editing sites, as well as thymidines homologous to these editing sites. The
  windows were extracted from 21 mitochondrial genomes of diverse angiosperms,
  whose C-to-U editing sites were identified by using deep RNA sequencing
  data. Identified editing sites are indicated by a fifth nucleotide:
  'E'.

* [_Lophophytum mirabile_ data](./datasets/lopho-data.tsv.gz). These data were used
  to evaluate the predictive performance of Deepred-Mt on unseen data, and was
  collected from the mitochondrial protein-coding sequences of a flowering
  plant called _Lophophytum mirabile_. The C-to-U editing sites of these
  sequences were [previously identified](https://doi.org/10.1111/nph.16926)
  with deep RNA sequencing.

Entries in both datasets are composed of the following fields:

  * Window ID.
  * Homolog ID.
  * Codon position: first/second/third codon position. Synonymous positions
    are indicated as a "fourth" codon position.
  * Upstream region of the nucleotide window.
  * Central position of the nucleotide window.
  * Downstream region of the nucleotide window.
  * Codon of the target position.
  * Editing extent of the target position: # of paired-end reads showing a T
    in the central position divided by the total number of paired-end reads
    aligned in that position.
  * Window label: 0/1.
  * PREP-Mt score.
  * PREPACT score (only for _Lophophytum mirabile_).
