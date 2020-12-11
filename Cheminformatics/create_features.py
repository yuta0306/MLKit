import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import ConvertToNumpyArray
import numpy as np
import pandas as pd

from typing import Any, AnyStr, Callable, Iterable, List, Tuple


def _create_fp(smile: AnyStr, radius: int=2, nBits: int=2048) -> np.ndarray:
    atorvation: rdkit.Chem.Mol = Chem.MolFromSmiles(smile)
    fingerprint = GetMorganFingerprintAsBitVect(atorvation, radius=radius, nBits=nBits)

    fp_array: np.ndarray = np.zeros((1, ))
    ConvertToNumpyArray(fingerprint, fp_array)

    return fp_array

def create_fps(smiles: Iterable[AnyStr], radius: int=2, nBits: int=2048, dtype: AnyStr='float64') -> np.ndarray:
    fps: np.ndarray = np.array([
        _create_fp(smile, radius=radius, nBits=nBits) for smile in smiles
    ], dtype=dtype)
    
    return fps

def fetch_mol(smile: AnyStr) -> rdkit.Chem.Mol:
    mol: rdkit.Chem.Mol = Chem.MolFromSmiles(smile)
    
    return mol

def fetch_features(smiles: Iterable[AnyStr]) -> Tuple[List[AnyStr], np.ndarray]:
    desc_list = Descriptors.descList
    funcnames = [t[0] for t in desc_list]
    funcs = [t[1] for t in desc_list]
    mols = [fetch_mol(smile) for smile in smiles]
    features = np.array([
        np.array([f(mol) for f in funcs])
            for mol in mols
    ])
    
    return funcnames, features