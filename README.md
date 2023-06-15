# Deepred-Mt: Deep Representation Learning for Predicting C-to-U RNA Editing in Plant Mitochondria

This repository contains the official implementation of Deepred-Mt, along with
instructions for reproducing results presented in
["_Deepred-Mt: Deep Representation Learning for Predicting C-to-U RNA Editing in Plant Mitochondria_"](https://www.sciencedirect.com/science/article/abs/pii/S0010482521004765),
by A. A. Edera, I. Small, D. H. Milone, and
M. V. Sanchez-Puerta. [Download PDF](https://sinc.unl.edu.ar/sinc-publications/2021/ESSM21/sinc_ESSM21.pdf).

In land plants, the editosome is a highly sophisticated molecular machine able
to convert post-transcriptionally cytidines into uridines (C-to-U) at highly
specific RNA positions called editing sites. This RNA editing seems to be partially governed by
_cis_ elements, which still remain recalcitrant to characterization.

Deepred-Mt is a novel neural network able to predict C-to-U editing sites in
angiosperm mitochondria. Given an RNA sequence, consisting of a central
cytidine flanked by 20 nucleotides on each side, Deepred-Mt scores how
probable its editing is.

<figure>
  <p align="center">
  <img src=fig/convolution.png alt="Convolution" width="500" style="vertical-align:middle"/>
  </p>
</figure>

The score is computed from complex _cis_ elements or motifs automatically
extracted from the flanking bases by a multi-layer convolutional neural
network, whose full architecture is schematically shown below.

<figure>
  <p align="center">
  <img src=fig/model-architecture.png alt="Deepred-Mt" width="600" style="vertical-align:middle"/>
  </p>
</figure>

## Submit RNA sequences for predictions

To submit RNA/DNA sequences for predicting their C-to-U editing sites with
Deepred-Mt, use the following link:

**[Submit sequences](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/05_fasta_submission.ipynb)**

**Note 1:** To be able to submit, you must be logged in with a Google Account
  (e.g., [Gmail](http://gmail.com)).

**Note 2:** If difficulties are experienced when submitting sequences, try to
  use [Google Chrome](https://www.google.com/chrome/) as the web browser.

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

Next, install Deepred-Mt from the sources

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

### Demo notebooks

The following notebooks reproduce experiments in the article.

| Description | Notebook |
|:------------|:--------:|
| Use Deepred-Mt on the command line to predict C-to-U editing sites from a given FASTA file|[<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/01_prediction_from_fasta.ipynb)|
| Compare the predictive performance of Deepred-Mt and state-of-the art methods for predicting editing sites| [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/02_reproduce_comparative_analysis.ipynb) |
| Train Deepred-Mt from scratch| [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/aedera/deepredmt/blob/main/notebooks/03_deepredmt_training.ipynb) |

## Data

Experiments used three datasets built from
[FASTA files](./data/fasta-files.tar.gz) containing mitochondrial protein
sequences from 21 plant species. These files have 'E' nucleotides to indicate
C-to-U editing sites, which were previously identified by using RNAseq data
publicly available in the
[European Nucleotide Archive](https://www.ebi.ac.uk/ena/browser/home).

| Dataset | Description |
|:-----|-------------|
|[Training data](./data/training-data.tsv.gz)| 41-bp nucleotide windows whose center positions are either unedited (C) or edited (E) cytidines. Nucleotide windows are labeled according to both the nucleotide in their central positions (0/C, 1/E) and their corresponding editing extents (a value ranging from 0 to 1)|
|[Task-related sequences](./data/task-related-sequences.tsv.gz)| Sequences used for the augmentation strategy proposed in the article. These sequences are 41-bp nucleotide windows whose center positions are thymidines homologous to one of the editing sites in the training data|
|[Control data](./data/control-data.tsv.gz)| Control data containing fake editing signal "GGCG" within the downstream regions of nucleotide windows that are labeled as 1 (edited)|

More information on the data format is provided [here](./data).


## Results

Deepred-Mt was compared to two state-of-the-art methods for predicting editing
sites: PREP-Mt and PREPACT. The following figure shows precision-recall curves
obtained from the predictions of each method. Deepred-Mt achieves the highest
F1 scores and the best areas under the curves (AUPRC) for two predictive
scenarios: one excluding synonymous sites (dashed lines) and other including
them (solid lines).

<p align="center">
<figure>
  <p align="center">
  <img src=fig/deepredmt-performance.png alt="Deepred-Mt performance" width="900" style="vertical-align:middle"/>
  </p>
</figure>
</p>

<table>
  <thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="2">Excluded</th>
    <th colspan="2">Included</th>
  </tr>
  <tr>
    <th>AUPRC</th>
    <th>F1</th>
    <th>AUPRC</th>
    <th>F1</th>
  </tr>
 </thead>
 <tbody>
  <tr>
    <td>PREPACT</td>
    <td>0.91</td>
    <td>0.89</td>
    <td>0.79</td>
    <td>0.82</td>
  </tr>
  <tr>
    <td>PREP-Mt</td>
    <td>0.88</td>
    <td>0.91</td>
    <td>0.76</td>
    <td>0.84</td>
  </tr>
  <tr>
    <td>Deepred-Mt</td>
    <td><b>0.96</b></td>
    <td><b>0.92</b></td>
    <td><b>0.91</b></td>
    <td><b>0.86</b></td>
  </tr>
 </tbody>
</table>


## Contributing

Contributions from anyone are welcome. You can start by adding a new entry [here](https://github.com/aedera/deepredmt/issues).


## License

Deepred-Mt is licensed under the MIT license. See [LICENSE](./LICENSE) for more details.
