# Nucleotide to integer
global _NT2ID
_NT2ID = {
    '-': -1,
    'N': -1,
    'M': -1,
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    'E': 4,
    'e': 5
}

from .train import fit
