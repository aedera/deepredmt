import os

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
    'U': 3,
    'E': 4,
    'e': 5
}

CANDIDATE_MODEL = '210520.tf'
current_path = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_path, './model')
global MODEL_FIN
MODEL_FIN = os.path.join(model_dir, CANDIDATE_MODEL)

from .train import fit
from .predict import predict
from .predict import predict_from_fasta
from .predict import get_vector_representations
from .project import project

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no tf warnings
