# Deepred-Mt

In land plants, the editosome is a highly sophisticated molecular machine able
to bind organellar RNA molecules post-transcriptionally. It converts cytidines
to uridines (C-to-U) at highly specific RNA positions called editing
sites. RNA editing is governed by _cis_ elements that remain recalcitrant to
characterization, limiting further advances on genetic engineering techniques
for RNA manipulation.

Deepred-Mt is a novel method to predict C-to-U editing sites in angiosperm
mitochondrial RNA. Given an RNA sequence consisting of a central cytidine
flanked by 20 nucleotides on each side, Deepred-Mt scores if the central
cytidine is edited. This score is computed from the sequence by extracting
sequence motifs with a convolutional approach.

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

After installing, you can run Deepred-Mt on the command line to predict C-to-U
editing sites for a desired FASTA file:

```bash
deepredmt data/seqs.fas
```

This searches the fasta file for all the cytidines and then predicts if they
are edited based on their surrounding nucleotides.

In addition, you can also use a plain text file containing nucleotide windows
of 41-bp as input for predictions:

```bash
deepredmt data/wins.txt
```

### Demo notebooks

* [Notebook 1](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/01_prediction_from_fasta.ipynb)
  shows how to use Deepred-Mt on the command line to predict C-to-U editing
  sites in a fasta file.

*
  [Notebook 2](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/02_reproduce_comparative_analysis.ipynb)
  reproduces results of the manuscript in which the predictive performance of
  Deepred-Mt is compared with those of state-of-the art methods for predicting
  editing sites.

* [Notebook 3](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/03_deepredmt_training.ipynb)
  shows how to train Deepred-Mt from scratch.

* [Notebook 4](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/04_single_sequence_submission.ipynb)
  can be used to submit DNA sequences to predict their editing sites with Deepred-Mt.

## Data

For our experiments, we use [FASTA files](./data/fasta-files.tar.gz)
containing mitochondrial protein-encoding sequences of 21 plant species. These
files have annotated editing sites, as 'E' nucleotides, which were previously
identified by using RNAseq data publicly available in the
[European Nucleotide Archive](https://www.ebi.ac.uk/ena/browser/home).

We use these FASTA files to construct three datasets:


* [Training data](./data/training-data.tsv.gz). It contains 41-bp nucleotide
  windows whose center positions are either unedited (C) or edited (E)
  cytidines. Nucleotide windows are labeled according to both the nucleotide
  in their central positions (0/C, 1/E) and their corresponding editing
  extents.

* [Task-related sequences](./data/task-related-sequences.tsv.gz). It contains
  the sequences derived from the task-related augmentation strategy proposed
  in the manuscript. These sequences are 41-bp nucleotide windows whose center
  positions are thymidines homologous to one of the editing sites in the
  training data.

* [Control data](./data/control-data.tsv.gz). This dataset was constructed
  from the training dataset. The fake editing signal "GGCG" was injected in
  the downstream regions of the nucleotide windows labeled as 1. Each time
  this fake signal was injected, one of its four nucleotides was mutated
  randomly with a probability inversely proportional to the editing extent of
  the sequence.

You can find more information on the format of these datasets in this
[README](./data) file.
