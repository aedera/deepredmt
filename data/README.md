## Datasets

The datasets have entries composed of the following fields:

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
  * [PREP-Mt](http://prep.unl.edu/) score.
  * [PREPACT](http://www.prepact.de/prepact-main.php) score (only for
    _Lophophytum mirabile_ dataset).

In the nucleotide windows, experimentally identified C-to-U editing sites are
indicated by a fifth nucleotide: 'E'.
