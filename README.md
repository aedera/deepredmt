# Deepred-Mt: Deep Representation Learning for Predicting C-to-U RNA Editing in Plant Mitochondria

In land plants, the editosome is a highly sophisticated molecular machine able
to bind organellar RNA molecules post-transcriptionally. It converts cytidines
to uridines (C-to-U) at highly specific RNA positions called editing
sites. RNA editing is governed by _cis_ elements that remain recalcitrant to
characterization, limiting further advances on genetic engineering techniques
for RNA manipulation.

Deepred-Mt is a neural network able to predict C-to-U editing sites from
mitochondrial RNA of angiosperms. Given an RNA sequence, consisting of a
central cytidine flanked by 20 nucleotides on each side, Deepred-Mt scores how
probable its editing is.

<figure>
  <p align="center">
  <img src=fig/convolution.png alt="Convolution" width="500" style="vertical-align:middle"/>
  </p>
</figure>

The score is computed from motifs automatically extracted from the flanking
bases by a multi-layer convolutional neural network, whose full architecture
is schematically shown below.

<figure>
  <p align="center">
  <img src=fig/model-architecture.png alt="Deepred-Mt" width="800" style="vertical-align:middle"/>
  </p>
</figure>

This repository contains the official implementation of Deepred-Mt, along with
instructions for reproducing results presented in
["_Deepred-Mt: Deep Representation Learning for Predicting C-to-U RNA Editing in Plant Mitochondria_"](https://www.sciencedirect.com/science/article/abs/pii/S0010482521004765),
by A. A. Edera, I. Small, D. H. Milone, and
M. V. Sanchez-Puerta. [Download PDF](https://sinc.unl.edu.ar/sinc-publications/2021/ESSM21/sinc_ESSM21.pdf).

## Submit RNA sequences for predictions

To submit RNA/DNA sequences for predicting their C-to-U editing sites with
Deepred-Mt, use the following link:

**[Submit sequences](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/05_fasta_submission.ipynb)**

**Note:** If difficulties are experimenting when submitting sequences, try to
  use [Google Chrome](https://www.google.com/chrome/) as a web browser.

## Installation

To install Deepred-Mt on your computer, the following dependencies must be
installed:

* [Python 3.7](https://www.python.org/)
* [pip](https://pip.pypa.io/en/stable/)
* [Git](https://git-scm.com/)
* [Conda](https://docs.conda.io/en/latest/)

First, create and activate a new Conda environment

```bash
conda create -n deepredmt python=3.7
conda activate deepredmt

```

Next, install Deepred-Mt on this environment

```bash
pip install -U "deepredmt @ git+https://github.com/aedera/deepredmt.git"
```

## Example usage

### Command line

Once installed, Deepred-Mt can be executed on the command line to predict
C-to-U editing sites from a desired FASTA
file. [Here](https://raw.githubusercontent.com/aedera/deepredmt/main/data/seqs.fas)
is an example FASTA file called `seqs.fas`:

```bash
deepredmt seqs.fas
```

This command extracts cytidines from the FASTA file to make predictions based
on their surrounding nucleotides.

Instead of a FASTA file, Deepred-Mt can also take in a plain text file containing
nucleotide windows of 41-bp. [Here](https://raw.githubusercontent.com/aedera/deepredmt/main/data/wins.txt)
is an example file called `wins.txt`:

```bash
deepredmt wins.txt
```

### Demo notebooks

The following notebooks reproduce experiments in the article.

| Notebook| Description |
|:-----|-------------|
| [Notebook 1](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/01_prediction_from_fasta.ipynb)| Show how to use Deepred-Mt on the command line to predict C-to-U editing sites from a given FASTA file|
| [Notebook 2](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/02_reproduce_comparative_analysis.ipynb) | Compare the predictive performance of Deepred-Mt with those of state-of-the art methods for predicting editing sites|
| [Notebook 3](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/03_deepredmt_training.ipynb) | Show how to train Deepred-Mt from scratch|

## Data

Experiments used [FASTA files](./data/fasta-files.tar.gz) containing
mitochondrial protein-encoding sequences from 21 plant species. These files
have 'E' nucleotides to indicate C-to-U editing sites, which were previously
identified by using RNAseq data publicly available in the
[European Nucleotide Archive](https://www.ebi.ac.uk/ena/browser/home).

These FASTA files were used to construct the following three datasets for
training and evaluating Deepred-Mt:

| Data | Description |
|:-----|-------------|
|[Training data](./data/training-data.tsv.gz)| 41-bp nucleotide windows whose center positions are either unedited (C) or edited (E) cytidines. Nucleotide windows are labeled according to both the nucleotide in their central positions (0/C, 1/E) and their corresponding editing extents (a value ranging from 0 to 1)|
|[Task-related sequences](./data/task-related-sequences.tsv.gz)| Sequences used for the augmentation strategy proposed in the article. These sequences are 41-bp nucleotide windows whose center positions are thymidines homologous to one of the editing sites in the training data|
|[Control data](./data/control-data.tsv.gz)| Control data containing fake editing signal "GGCG" within the downstream regions of nucleotide windows that are labeled as 1 (edited)|

More information on the format of these data is provided in this [file](./data).


## Contributing

Contributions from anyone are welcome. You can start by adding a new entry [here](https://github.com/aedera/deepredmt/issues).


## License

Deepred-Mt is licensed under the MIT license. See [LICENSE](./LICENSE) for more details.
