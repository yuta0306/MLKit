import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, Draw
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import ConvertToNumpyArray
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import load_img, img_to_array

from typing import Any, AnyStr, Callable, Iterable, List, Tuple
import os


def _create_fp(smile: AnyStr, radius: int=2, nBits: int=2048) -> np.ndarray:
    atorvation: rdkit.Chem.Mol = Chem.MolFromSmiles(smile)
    fingerprint = GetMorganFingerprintAsBitVect(atorvation, radius=radius, nBits=nBits)

    fp_array: np.ndarray = np.zeros((1, ))
    ConvertToNumpyArray(fingerprint, fp_array)

    return fp_array

def create_fps(smiles: Iterable[AnyStr], radius: int=2, nBits: int=2048, dtype: AnyStr='float64') -> np.ndarray:
    fps: np.ndarray = np.array([
        _create_fp(smile, radius=radius, nBits=nBits) for smile in tqdm(smiles)
    ], dtype=dtype)
    
    return fps

def fetch_mol(smile: AnyStr) -> rdkit.Chem.Mol:
    mol: rdkit.Chem.Mol = Chem.MolFromSmiles(smile)
    
    return mol

def fetch_features(smiles: Iterable[AnyStr]) -> Tuple[List[AnyStr], np.ndarray]:
    desc_list = Descriptors.descList
    funcnames: List[AnyStr] = [t[0] for t in desc_list]
    funcs: List[Callable] = [t[1] for t in desc_list]
    mols = [fetch_mol(smile) for smile in smiles]
    features: np.ndarray[Any] = np.array([
        np.array([f(mol) for f in funcs])
            for mol in tqdm(mols)
    ])
    
    return funcnames, features

def generate_imgs(smiles: List[AnyStr], target_dir: AnyStr='.') ->List[AnyStr]:
    images: List[AnyStr] = []
    if target_dir[-1] == '/':
        target_dir = target_dir[:-1]
    os.makedirs(target_dir, exist_ok=True)
    for i, smile in enumerate(smiles):
        mol = fetch_mol(smile)
        Draw.MolToFile(mol, f'{target_dir}/mol_{i}.png')
        images.append(f'{target_dir}/mol_{i}.png')
        
    return images

def images_to_array(images_path: List[AnyStr], color_mode: AnyStr='rgb', scaling: bool=False) -> np.ndarray:
    images: np.ndarray = np.array([
        img_to_array(load_img(path, color_mode=color_mode))
            for path in tqdm(images_path)
    ])
    
    if scaling:
        images = images / 255
    
    return images