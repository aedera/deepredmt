## Datasets

The datasets have entries composed of the following tab-separated
fields:

  * Field 01. Window ID.
  * Field 02. Homolog ID.
  * Field 03. Codon position: first/second/third codon
    position. Synonymous positions are indicated as a "fourth" codon
    position.
  * Field 04. Upstream region of the nucleotide window.
  * Field 05. Central position of the nucleotide window.
  * Field 06. Downstream region of the nucleotide window.
  * Field 07. Codon of the target position.
  * Field 08. Editing extent of the target position: # of paired-end
    reads showing a T in the central position divided by the total
    number of paired-end reads aligned in that position.
  * Field 09. Window label: 0/1.
  * Field 10. [PREP-Mt](http://prep.unl.edu/) score.
  * Field 11. [PREPACT](http://www.prepact.de/prepact-main.php) score
    (only for _Lophophytum mirabile_ dataset).

In the nucleotide windows, experimentally identified C-to-U editing
sites are indicated by a fifth nucleotide: 'E'.
