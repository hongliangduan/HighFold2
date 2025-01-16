from __future__ import annotations

import os

ENV = {"TF_FORCE_UNIFIED_MEMORY": "1", "XLA_PYTHON_CLIENT_MEM_FRACTION": "4.0"}
for k, v in ENV.items():
    if k not in os.environ: os.environ[k] = v

import warnings
from Bio import BiopythonDeprecationWarning  # what can possibly go wrong...

warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)

import json
import logging
import math
import random
import sys
import time
import zipfile
import shutil
import pickle

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from io import StringIO

import importlib_metadata
import numpy as np
import pandas
import haiku as hk

try:
    import alphafold
except ModuleNotFoundError:
    raise RuntimeError(
        "\n\nalphafold is not installed. Please run `pip install colabfold[alphafold]`\n"
    )

from alphafold.common import protein, residue_constants

# delay imports of tensorflow, jax and numpy
# loading these for type checking only can take around 10 seconds just to show a CLI usage message
if TYPE_CHECKING:
    import haiku
    from alphafold.model import model
    from numpy import ndarray

from alphafold.common.protein import Protein
from alphafold.data import (
    feature_processing,
    msa_pairing,
    pipeline,
    pipeline_multimer,
    templates,
)
from alphafold.data.tools import hhsearch
from colabfold.citations import write_bibtex
from colabfold.download import default_data_dir, download_alphafold_params
from colabfold.utils import (
    ACCEPT_DEFAULT_TERMS,
    DEFAULT_API_SERVER,
    NO_GPU_FOUND,
    CIF_REVISION_DATE,
    get_commit,
    safe_filename,
    setup_logging,
    CFMMCIFIO,
)

from Bio.PDB import MMCIFParser, PDBParser, MMCIF2Dict
from Bio.PDB.PDBIO import Select

# logging settings
logger = logging.getLogger(__name__)
import jax
import jax.numpy as jnp

logging.getLogger('jax._src.lib.xla_bridge').addFilter(lambda _: False)
logging.getLogger('parmed').setLevel(logging.WARNING)

# zc
from rdkit import Chem
import copy
class PoolLayer(hk.Module):
    def __init__(self):
        super(PoolLayer, self).__init__()
        # self.bond_qury = hk.Sequential([
        #     hk.Linear(6),
        #     jax.nn.relu,
        #     hk.Linear(6),
        #     jax.nn.relu,
        #     hk.Linear(6),
        # ])
        # self.bond_key = hk.Sequential([
        #     hk.Linear(6),
        #     jax.nn.relu,
        #     hk.Linear(6),
        #     jax.nn.relu,
        #     hk.Linear(6),
        # ])
        # self.bond_value = hk.Sequential([
        #     hk.Linear(6),
        #     jax.nn.relu,
        #     hk.Linear(6),
        #     jax.nn.relu,
        #     hk.Linear(6),
        # ])
        self.linear1 = hk.Sequential([
            hk.Linear(6),
            jax.nn.relu,
            hk.Linear(6),
            jax.nn.relu,
            hk.Linear(1),
        ])
        self.linear2 = hk.Sequential([
            hk.Linear(1),
            jax.nn.relu,
            hk.Linear(64),
            jax.nn.relu,
            hk.Linear(128),
        ])
        self.atom_qury = hk.Sequential([
            hk.Linear(11),
            jax.nn.relu,
            hk.Linear(11),
            jax.nn.relu,
            hk.Linear(21),
        ])
        self.atom_key = hk.Sequential([
            hk.Linear(11),
            jax.nn.relu,
            hk.Linear(11),
            jax.nn.relu,
            hk.Linear(21),
        ])
        self.atom_value = hk.Sequential([
            hk.Linear(11),
            jax.nn.relu,
            hk.Linear(11),
            jax.nn.relu,
            hk.Linear(21),
        ])
        # self.linear3 = hk.Sequential([
        #     hk.Linear(11),
        #     jax.nn.relu,
        #     hk.Linear(11),
        #     jax.nn.relu,
        #     hk.Linear(21),
        # ])

    def __call__(self, bond_feat, atom_feat, M):
        F = (1.0 - M + 1e-6) / (M - 1e-6)
        
        # N, N, n = bond_feat.shape
        # bond_feat = jnp.reshape(bond_feat, (N*N, 6))
        # bond_qury = self.bond_qury(bond_feat) # [N*N, 6]
        # bond_key = self.bond_key(bond_feat) # [N*N, 6]
        # bond_value = self.bond_value(bond_feat) # [N*N, 6]
        # score = jax.nn.softmax(jnp.matmul(bond_qury, bond_key.T) / jnp.sqrt(6)) # [N*N, N*N]
        # bond = jnp.matmul(score, bond_value) # [N*N, 6]
        # bond = jnp.reshape(bond, (N, N, 6))
        
        bond = self.linear1(bond_feat) # [N, N, 1]
        Ms_bond1 = jax.nn.softmax(jnp.transpose(bond, (0, 2, 1)) + F[:, :, None], axis=0)  # [N, Nr, N]
        bond_pool = jnp.matmul(bond.transpose(0, 2, 1), Ms_bond1.transpose(0, 2, 1))  # [N, 1, Nr]

        Ms_bond2 = jax.nn.softmax(bond_pool + F[:, :, None], axis=0) # [N, Nr, Nr]
        bond_pool = jnp.matmul(bond_pool.transpose(2, 1, 0), Ms_bond2.transpose(1, 0, 2)) # [Nr, 1, Nr]
        bond_pool = self.linear2(bond_pool.transpose(0, 2, 1)) # [Nr, Nr, 1]
        # bond_pool = jnp.squeeze(bond_pool, axis=2)


        atom_qury = self.atom_qury(atom_feat) # [N, 21]
        atom_key = self.atom_key(atom_feat) # [N, 21]
        atom_value = self.atom_value(atom_feat) # [N, 21]
        score = jax.nn.softmax(jnp.matmul(atom_qury, atom_key.T) / jnp.sqrt(11)) # [N, N]
        atom = jnp.matmul(score, atom_value) # [N, 21]
        # atom = self.linear3(atom)
        Ms_atom = jax.nn.softmax(atom[:, None, :] + F[:, :, None], axis=0) # [N,Nr,21]
        atom_pool = jnp.matmul(atom.T[:, None, :], Ms_atom.transpose(2, 0, 1)) # [21,1,Nr]
        atom_pool = jnp.squeeze(atom_pool, axis=1).T
        return bond_pool, atom_pool

def pool_fn(bond_feat, atom_feat, M):
    model = PoolLayer()
    return model(bond_feat, atom_feat, M)

smiles_dict = {'GLY': 'NCC(=O)',
 'THR': 'N[C@@]([H])([C@]([H])(O)C)C(=O)',
 'LYS': 'N[C@@]([H])(CCCCN)C(=O)',
 'VAL': 'N[C@@]([H])(C(C)C)C(=O)',
 'HIS': 'N[C@@]([H])(CC1=CN=C-N1)C(=O)',
 'ARG': 'N[C@@]([H])(CCCNC(=N)N)C(=O)',
 'ALA': 'N[C@@]([H])(C)C(=O)',
 'LEU': 'N[C@@]([H])(CC(C)C)C(=O)',
 'ILE': 'N[C@@]([H])([C@]([H])(CC)C)C(=O)',
 'MET': 'N[C@@]([H])(CCSC)C(=O)',
 'PHE': 'N[C@@]([H])(Cc1ccccc1)C(=O)',
 'TRP': 'N[C@@]([H])(CC(=CN2)C1=C2C=CC=C1)C(=O)',
 'PRO': 'N1[C@@]([H])(CCC1)C(=O)',
 'SER': 'N[C@@]([H])(CO)C(=O)',
 'CYS': 'N[C@@]([H])(CS)C(=O)',
 'TYR': 'N[C@@]([H])(Cc1ccc(O)cc1)C(=O)',
 'ASN': 'N[C@@]([H])(CC(=O)N)C(=O)',
 'GLN': 'N[C@@]([H])(CCC(=O)N)C(=O)',
 'ASP': 'N[C@@]([H])(CC(=O)O)C(=O)',
 'GLU': 'N[C@@]([H])(CCC(=O)O)C(=O)',
 'CSO': 'N[C@@]([H])(CSO)C(=O)',
 'NVA': 'N[C@@]([H])(CCC)C(=O)',
 'NLE': 'N[C@@]([H])(CCCC)C(=O)',
 'AIB': 'NC(C)(C)C(=O)',
 'MLZ': 'N[C@@]([H])(CCCCNC)C(=O)',
 'PTR': 'N[C@@]([H])(Cc1ccc(OP(O)(=O)O)cc1)C(=O)',
 'HYP': 'N1[C@@]([H])(CC(=O)C1)C(=O)',
 'SEP': 'N[C@@]([H])(COP(O)(=O)O)C(=O)',
 'TPO': 'N[C@@]([H])(C(C)OP(O)(=O)O)C(=O)',
 'MLY': 'N[C@@]([H])(CCCCN(C)C)C(=O)',
 'ALY': 'N[C@@]([H])(CCCCNC(=O)C)C(=O)',
 'ESC': 'N[C@@]([H])(CCSCC)C(=O)',
 'PCA': 'N1[C@@]([H])(CCC1(=O))C(=O)',
 'M3L': 'N[C@@]([H])(CCCC[N+](C)(C)C)C(=O)',
 'ABA': 'N[C@@]([H])(CC)C(=O)',
 'DA2': 'N[C@@]([H])(CCCNC(N)N(C)C)C(=O)',
 '2MR': 'N[C@@]([H])(CCCNC(NC)NC)C(=O)',
 'ALC': 'N[C@@]([H])(C(CCC1)(CC1)C)C(=O)',
 'HZP': 'N1[C@@]([H])(CC(=O)C1)C(=O)',
 'HOX': 'N[C@@]([H])(Cc1ccc(N)cc1)C(=O)',
 'PRK': 'N[C@@]([H])(CCCCNC(=O)CC)C(=O)',
 'DAL': 'N[C@]([H])(C)C(=O)',
 'DHI': 'N[C@]([H])(CC1=CN=C-N1)C(=O)',
 'DPR': 'N1[C@]([H])(CCC1)C(=O)',
 'DVA': 'N[C@]([H])(C(C)C)C(=O)',
 'DSN': 'N[C@]([H])(CO)C(=O)',
 'DGN': 'N[C@]([H])(CCC(=O)N)C(=O)',
 'DTY': 'N[C@]([H])(Cc1ccc(O)cc1)C(=O)',
 'DAS': 'N[C@]([H])(CC(=O)O)C(=O)',
 'DLE': 'N[C@]([H])(CC(C)C)C(=O)',
 'DTR': 'N[C@]([H])(CC(=CN2)C1=C2C=CC=C1)C(=O)',
 'DPN': 'N[C@]([H])(Cc1ccccc1)C(=O)',
 'DGL': 'N[C@]([H])(CCC(=O)O)C(=O)',
 'DAR': 'N[C@]([H])(CCCNC(=N)N)C(=O)',
 'DIL': 'N[C@]([H])([C@]([H])(CC)C)C(=O)',
 'MSE': 'N[C@@]([H])(CC[Se]C)C(=O)',
 'ORN': 'N[C@@]([H])(CCCN)C(=O)',
 'NH2': 'N',
 }

def bonds_from_smiles(smiles_string, atom_encoding):
    """Get all bonds from the smiles
    """
    m = Chem.MolFromSmiles(smiles_string)

    bond_encoding = {'SINGLE':1, 'DOUBLE':2, 'TRIPLE':3, 'AROMATIC':4, 'IONIC':5}

    #Go through the smiles and assign the atom types and a bond matrix
    atoms = []
    atom_types = []
    num_atoms = len(m.GetAtoms())
    bond_matrix = np.zeros((num_atoms, num_atoms))


    #Get the atom types
    for atom in m.GetAtoms():
        atoms.append(atom.GetSymbol())
        atom_types.append(atom_encoding.get(atom.GetSymbol(),10))
        #Get neighbours and assign bonds
        for nb in atom.GetNeighbors():
            for bond in nb.GetBonds():
                bond_type = bond_encoding.get(str(bond.GetBondType()), 6)
                si = bond.GetBeginAtomIdx()
                ei = bond.GetEndAtomIdx()
                bond_matrix[si,ei] = bond_type
                bond_matrix[ei,si] = bond_type

    #Get a distance matrix
    #Add Hs
    # m = Chem.AddHs(m)
    #Embed in 3D
    # AllChem.EmbedMolecule(m, maxAttempts=500)
    #Remove Hs to fit other dimensions (will cause error if mismatch on mult with has_bond)
    # m = Chem.RemoveHs(m)
    # D=AllChem.Get3DDistanceMatrix(m)
    #Get the bond positions
    has_bond = copy.deepcopy(bond_matrix)
    has_bond[has_bond>0]=1

    # return np.array(atom_types), np.array(atoms), bond_matrix, D*has_bond, has_bond
    return np.array(atom_types), np.array(atoms), bond_matrix, has_bond

def get_atom_features(smiles_string):

    input_smiles = smiles_string
    #Atom encoding - no hydrogens
    atom_encoding = {'B':0, 'C':1, 'F':2, 'I':3, 'N':4, 'O':5, 'P':6, 'S':7,'Br':8, 'Cl':9, #Individual encoding
                    'As':10, 'Co':10, 'Fe':10, 'Mg':10, 'Pt':10, 'Rh':10, 'Ru':10, 'Se':10, 'Si':10, 'Te':10, 'V':10, 'Zn':10 #Joint (rare)
                    }

    #Get the atom types and bonds
    # atom_types, atoms, bond_types, bond_lengths, bond_mask = bonds_from_smiles(input_smiles, atom_encoding)
    atom_types, atoms, bond_types, bond_mask = bonds_from_smiles(input_smiles, atom_encoding)

    ligand_feats = {}
    ligand_feats['atoms'] = atoms
    ligand_feats['atom_types'] = atom_types
    ligand_feats['bond_types'] = bond_types
    # ligand_inp_feats['bond_lengths'] = bond_lengths
    ligand_feats['bond_mask'] = bond_mask
    return ligand_feats

def get_bond_feat(ligand_feats):
    atom_len = len(ligand_feats['atoms'])
    # bond_feat = np.zeros((atom_len,atom_len,6))
    # bond_feat[:,:,:5] = (np.eye(6)[np.array(ligand_feats['bond_types'],dtype=int)])[:,:,1:]
    # bond_feat[:,:,5] = ligand_feats['bond_mask']
    # or this way
    bond_feat = (np.eye(6)[np.array(ligand_feats['bond_types'],dtype=int)])
    return bond_feat
def get_atom_feat(ligand_feats):
    atom_types = ligand_feats['atom_types']
    atom_feat = np.zeros((atom_types.size, 11))
    atom_feat[np.arange(atom_types.size), atom_types] = 1
    return atom_feat
    

def get_mask(ligand_feats):
    array = ligand_feats['atoms']
    split_indices = [0]
    sequence = ['C', 'O', 'N']
    seq_len = len(sequence)
    i = 0

    while i <= len(array) - seq_len:
        if list(array[i:i + seq_len]) == sequence and np.count_nonzero(ligand_feats['bond_mask'][i+seq_len-1]) != 1:
            split_indices.append(i + seq_len - 1)
            i += seq_len
        else:
            i += 1

    new_array = np.zeros(len(array), dtype=int)
    for i in range(len(split_indices)):
        if i == len(split_indices) - 1:
            start = split_indices[i]
            new_array[start:] = i
        else:
            start = split_indices[i]
            end = split_indices[i + 1]
            new_array[start:end] = i

    num_classes = np.max(new_array) + 1
    one_hot_encoded = np.eye(num_classes)[new_array]
    return one_hot_encoded
#zc

def patch_openmm():
    from simtk.openmm import app
    from simtk.unit import nanometers, sqrt

    # applied https://raw.githubusercontent.com/deepmind/alphafold/main/docker/openmm.patch
    # to OpenMM 7.5.1 (see PR https://github.com/openmm/openmm/pull/3203)
    # patch is licensed under CC-0
    # OpenMM is licensed under MIT and LGPL
    # fmt: off
    def createDisulfideBonds(self, positions):
        def isCyx(res):
            names = [atom.name for atom in res._atoms]
            return 'SG' in names and 'HG' not in names

        # This function is used to prevent multiple di-sulfide bonds from being
        # assigned to a given atom.
        def isDisulfideBonded(atom):
            for b in self._bonds:
                if (atom in b and b[0].name == 'SG' and
                        b[1].name == 'SG'):
                    return True

            return False

        cyx = [res for res in self.residues() if res.name == 'CYS' and isCyx(res)]
        atomNames = [[atom.name for atom in res._atoms] for res in cyx]
        for i in range(len(cyx)):
            sg1 = cyx[i]._atoms[atomNames[i].index('SG')]
            pos1 = positions[sg1.index]
            candidate_distance, candidate_atom = 0.3 * nanometers, None
            for j in range(i):
                sg2 = cyx[j]._atoms[atomNames[j].index('SG')]
                pos2 = positions[sg2.index]
                delta = [x - y for (x, y) in zip(pos1, pos2)]
                distance = sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])
                if distance < candidate_distance and not isDisulfideBonded(sg2):
                    candidate_distance = distance
                    candidate_atom = sg2
            # Assign bond to closest pair.
            if candidate_atom:
                self.addBond(sg1, candidate_atom)

    # fmt: on
    app.Topology.createDisulfideBonds = createDisulfideBonds


def mk_mock_template(
        query_sequence: Union[List[str], str], num_temp: int = 1
) -> Dict[str, Any]:
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
    output_confidence_scores = np.full(ln, 1.0)

    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_confidence_scores": np.tile(
            output_confidence_scores[None], [num_temp, 1]
        ),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_release_date": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    return template_features


def mk_template(
        a3m_lines: str, template_path: str, query_sequence: str
) -> Dict[str, Any]:
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_path,
        max_template_date="2100-01-01",
        max_hits=20,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[f"{template_path}/pdb70"]
    )

    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hhsearch_hits
    )
    return dict(templates_result.features)


def validate_and_fix_mmcif(cif_file: Path):
    """validate presence of _entity_poly_seq in cif file and add revision_date if missing"""
    # check that required poly_seq and revision_date fields are present
    cif_dict = MMCIF2Dict.MMCIF2Dict(cif_file)
    required = [
        "_chem_comp.id",
        "_chem_comp.type",
        "_struct_asym.id",
        "_struct_asym.entity_id",
        "_entity_poly_seq.mon_id",
    ]
    for r in required:
        if r not in cif_dict:
            raise ValueError(f"mmCIF file {cif_file} is missing required field {r}.")
    if "_pdbx_audit_revision_history.revision_date" not in cif_dict:
        logger.info(
            f"Adding missing field revision_date to {cif_file}. Backing up original file to {cif_file}.bak."
        )
        shutil.copy2(cif_file, str(cif_file) + ".bak")
        with open(cif_file, "a") as f:
            f.write(CIF_REVISION_DATE)


modified_mapping = {
    "MSE": "MET", "MLY": "LYS", "FME": "MET", "HYP": "PRO",
    "TPO": "THR", "CSO": "CYS", "SEP": "SER", "M3L": "LYS",
    "HSK": "HIS", "SAC": "SER", "PCA": "GLU", "DAL": "ALA",
    "CME": "CYS", "CSD": "CYS", "OCS": "CYS", "DPR": "PRO",
    "B3K": "LYS", "ALY": "LYS", "YCM": "CYS", "MLZ": "LYS",
    "4BF": "TYR", "KCX": "LYS", "B3E": "GLU", "B3D": "ASP",
    "HZP": "PRO", "CSX": "CYS", "BAL": "ALA", "HIC": "HIS",
    "DBZ": "ALA", "DCY": "CYS", "DVA": "VAL", "NLE": "LEU",
    "SMC": "CYS", "AGM": "ARG", "B3A": "ALA", "DAS": "ASP",
    "DLY": "LYS", "DSN": "SER", "DTH": "THR", "GL3": "GLY",
    "HY3": "PRO", "LLP": "LYS", "MGN": "GLN", "MHS": "HIS",
    "TRQ": "TRP", "B3Y": "TYR", "PHI": "PHE", "PTR": "TYR",
    "TYS": "TYR", "IAS": "ASP", "GPL": "LYS", "KYN": "TRP",
    "CSD": "CYS", "SEC": "CYS"
}


class ReplaceOrRemoveHetatmSelect(Select):
    def accept_residue(self, residue):
        hetfield, _, _ = residue.get_id()
        if hetfield != " ":
            if residue.resname in modified_mapping:
                # set unmodified resname
                residue.resname = modified_mapping[residue.resname]
                # clear hetatm flag
                residue._id = (" ", residue._id[1], " ")
                t = residue.full_id
                residue.full_id = (t[0], t[1], t[2], residue._id)
                return 1
            return 0
        else:
            return 1


def convert_pdb_to_mmcif(pdb_file: Path):
    """convert existing pdb files into mmcif with the required poly_seq and revision_date"""
    i = pdb_file.stem
    cif_file = pdb_file.parent.joinpath(f"{i}.cif")
    if cif_file.is_file():
        return
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(i, pdb_file)
    cif_io = CFMMCIFIO()
    cif_io.set_structure(structure)
    cif_io.save(str(cif_file), ReplaceOrRemoveHetatmSelect())


def mk_hhsearch_db(template_dir: str):
    template_path = Path(template_dir)

    cif_files = template_path.glob("*.cif")
    for cif_file in cif_files:
        validate_and_fix_mmcif(cif_file)

    pdb_files = template_path.glob("*.pdb")
    for pdb_file in pdb_files:
        convert_pdb_to_mmcif(pdb_file)

    pdb70_db_files = template_path.glob("pdb70*")
    for f in pdb70_db_files:
        os.remove(f)

    with open(template_path.joinpath("pdb70_a3m.ffdata"), "w") as a3m, open(
            template_path.joinpath("pdb70_cs219.ffindex"), "w"
    ) as cs219_index, open(
        template_path.joinpath("pdb70_a3m.ffindex"), "w"
    ) as a3m_index, open(
        template_path.joinpath("pdb70_cs219.ffdata"), "w"
    ) as cs219:
        n = 1000000
        index_offset = 0
        cif_files = template_path.glob("*.cif")
        for cif_file in cif_files:
            with open(cif_file) as f:
                cif_string = f.read()
            cif_fh = StringIO(cif_string)
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("none", cif_fh)
            models = list(structure.get_models())
            if len(models) != 1:
                raise ValueError(
                    f"Only single model PDBs are supported. Found {len(models)} models."
                )
            model = models[0]
            for chain in model:
                amino_acid_res = []
                for res in chain:
                    if res.id[2] != " ":
                        raise ValueError(
                            f"PDB contains an insertion code at chain {chain.id} and residue "
                            f"index {res.id[1]}. These are not supported."
                        )
                    amino_acid_res.append(
                        residue_constants.restype_3to1.get(res.resname, "X")
                    )

                protein_str = "".join(amino_acid_res)
                a3m_str = f">{cif_file.stem}_{chain.id}\n{protein_str}\n\0"
                a3m_str_len = len(a3m_str)
                a3m_index.write(f"{n}\t{index_offset}\t{a3m_str_len}\n")
                cs219_index.write(f"{n}\t{index_offset}\t{len(protein_str)}\n")
                index_offset += a3m_str_len
                a3m.write(a3m_str)
                cs219.write("\n\0")
                n += 1


def pad_input(
        input_features: model.features.FeatureDict,
        model_runner: model.RunModel,
        model_name: str,
        pad_len: int,
        use_templates: bool,
) -> model.features.FeatureDict:
    from colabfold.alphafold.msa import make_fixed_size

    model_config = model_runner.config
    eval_cfg = model_config.data.eval
    crop_feats = {k: [None] + v for k, v in dict(eval_cfg.feat).items()}

    max_msa_clusters = eval_cfg.max_msa_clusters
    max_extra_msa = model_config.data.common.max_extra_msa
    # templates models
    if (model_name == "model_1" or model_name == "model_2") and use_templates:
        pad_msa_clusters = max_msa_clusters - eval_cfg.max_templates
    else:
        pad_msa_clusters = max_msa_clusters

    max_msa_clusters = pad_msa_clusters

    # let's try pad (num_res + X)
    input_fix = make_fixed_size(
        input_features,
        crop_feats,
        msa_cluster_size=max_msa_clusters,  # true_msa (4, 512, 68)
        extra_msa_size=max_extra_msa,  # extra_msa (4, 5120, 68)
        num_res=pad_len,  # aatype (4, 68)
        num_templates=4,
    )  # template_mask (4, 4) second value
    return input_fix


def relax_me(dist_cst_ss=[], pdb_filename=None, pdb_lines=None, pdb_obj=None, use_gpu=False, disulf=None, aa_names=None, nc=None,cp=None):
    if "relax" not in dir():
        patch_openmm()
        from alphafold.common import residue_constants
        from alphafold.relax import relax

    if pdb_obj is None:
        if pdb_lines is None:
            pdb_lines = Path(pdb_filename).read_text()
        pdb_obj = protein.from_pdb_string(pdb_lines)

    amber_relaxer = relax.AmberRelaxation(
        dist_cst_ss=dist_cst_ss,
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=3,
        use_gpu=use_gpu,
        disulf=disulf,
        aa_names=aa_names,
        nc=nc,
        cp=cp)

    relaxed_pdb_lines, _, _ = amber_relaxer.process(prot=pdb_obj)
    return relaxed_pdb_lines


class file_manager:
    def __init__(self, prefix: str, result_dir: Path):
        self.prefix = prefix
        self.result_dir = result_dir
        self.tag = None
        self.files = {}

    def get(self, x: str, ext: str) -> Path:
        if self.tag not in self.files:
            self.files[self.tag] = []
        file = self.result_dir.joinpath(f"{self.prefix}_{x}_{self.tag}.{ext}")
        self.files[self.tag].append([x, ext, file])
        return file

    def set_tag(self, tag):
        self.tag = tag


def predict_structure(
        dist_cst_ss: List[List[int]],
        index_ss: Dict,
        prefix: str,
        result_dir: Path,
        feature_dict: Dict[str, Any],
        is_complex: bool,
        use_templates: bool,
        sequences_lengths: List[int],
        pad_len: int,
        model_type: str,
        model_runner_and_params: List[Tuple[str, model.RunModel, haiku.Params]],
        num_relax: int = 0,
        rank_by: str = "auto",
        random_seed: int = 0,
        num_seeds: int = 1,
        stop_at_score: float = 100,
        prediction_callback: Callable[[Any, Any, Any, Any, Any], Any] = None,
        use_gpu_relax: bool = False,
        save_all: bool = False,
        save_single_representations: bool = False,
        save_pair_representations: bool = False,
        save_recycles: bool = False,
        flag_nc: list = None,
        cp: list = None,
):
    """Predicts structure using AlphaFold for the given sequence."""

    mean_scores = []
    conf = []
    unrelaxed_pdb_lines = []
    prediction_times = []
    model_names = []
    files = file_manager(prefix, result_dir)
    seq_len = sum(sequences_lengths)

    # iterate through random seeds
    for seed_num, seed in enumerate(range(random_seed, random_seed + num_seeds)):

        # iterate through models
        for model_num, (model_name, model_runner, params) in enumerate(model_runner_and_params):
            pool_params, params = hk.data_structures.partition(lambda m, n, p: m[:9] != "alphafold", params)
            pool = hk.transform(pool_fn, apply_rng=True)
            indices = [i for i, arr in enumerate(feature_dict['chain_length']) if arr < 50]
            ligand_feats_dict = {}
            for index in indices:
                end = 0
                for i in range(index+1):
                    end += feature_dict['chain_length'][i]
                start = end - feature_dict['chain_length'][index]
                pep_index = feature_dict['aatype']
                pep_index = feature_dict['aatype'][start:end]
                pep_chara = [residue_constants.resnames[i] for i in pep_index]
                ligand_smiles = ''
                for i in pep_chara:
                    ligand_smiles += smiles_dict[i]
                if pep_chara[-1] != 'NH2':
                    ligand_smiles += 'O'
                ligand_feats = get_atom_features(ligand_smiles)
                bond_feat = get_bond_feat(ligand_feats)
                atom_feat = get_atom_feat(ligand_feats)
                mask = get_mask(ligand_feats)
                if pep_chara[-1] == 'NH2':
                    new_mask = np.zeros((mask.shape[0], mask.shape[1] + 1))
                    new_mask[:, :mask.shape[1]] = mask
                    new_mask[-1, -1] = 1
                    new_mask[-1, -2] = 0
                    mask = new_mask
                ligand_feats_dict[index] = {'atom_feat':atom_feat, 'bond_feat':bond_feat, 'mask':mask, 'start_and_end':[start, end]}
            feature_dict['ligand_feats_dict'] = ligand_feats_dict
            rng = jax.random.PRNGKey(random_seed)
            L = len(feature_dict['aatype'])
            bond_rep_plus_init = jnp.zeros((L, L, 128), dtype = 'bfloat16')
            atom_rep_plus_init = jnp.zeros((L, 21), dtype = 'bfloat16')
            bond_rep_plus = bond_rep_plus_init
            atom_rep_plus = atom_rep_plus_init
            for key in feature_dict['ligand_feats_dict']:
                start, end = feature_dict['ligand_feats_dict'][key]['start_and_end']
                bond_feat = feature_dict['ligand_feats_dict'][key]['bond_feat']
                atom_feat = feature_dict['ligand_feats_dict'][key]['atom_feat']
                mask = feature_dict['ligand_feats_dict'][key]['mask']
                bond_rep, atom_rep = jax.jit(pool.apply)(pool_params, rng, bond_feat, atom_feat, mask)
                bond_rep_plus = bond_rep_plus.at[start:end,start:end,:].set(bond_rep.astype('bfloat16'))
                # bond_rep_plus = bound_rep_plus_init.at[start:end,start:end,0].set(bond_rep.astype('bfloat16'))
                atom_rep_plus = atom_rep_plus.at[start:end,:].set(atom_rep.astype('bfloat16'))
            feature_dict['bond_rep_plus'] = bond_rep_plus
            feature_dict['atom_rep_plus'] = atom_rep_plus
            del feature_dict['ligand_feats_dict']


            # swap params to avoid recompiling
            model_runner.params = params

            #########################
            # process input features
            #########################
            if "multimer" in model_type:
                if model_num == 0 and seed_num == 0:
                    # TODO: add pad_input_mulitmer()
                    input_features = feature_dict
                    input_features["asym_id"] = input_features["asym_id"] - input_features["asym_id"][..., 0]
                    # print(input_features.keys())
            else:
                if model_num == 0:
                    input_features = model_runner.process_features(feature_dict, random_seed=seed)
                    r = input_features["aatype"].shape[0]
                    input_features["asym_id"] = np.tile(feature_dict["asym_id"], r).reshape(r, -1)
                    if seq_len < pad_len:
                        input_features = pad_input(input_features, model_runner,
                                                   model_name, pad_len, use_templates)
                        logger.info(f"Padding length to {pad_len}")

            tag = f"{model_type}_{model_name}_seed_{seed:03d}"
            model_names.append(tag)
            files.set_tag(tag)

            ########################
            # predict
            ########################
            start = time.time()

            # monitor intermediate results
            def callback(result, recycles):
                if recycles == 0: result.pop("tol", None)
                if not is_complex: result.pop("iptm", None)
                print_line = ""
                for x, y in [["mean_plddt", "pLDDT"], ["ptm", "pTM"], ["iptm", "ipTM"], ["tol", "tol"]]:
                    if x in result:
                        print_line += f" {y}={result[x]:.3g}"
                logger.info(f"{tag} recycle={recycles}{print_line}")

                if save_recycles:
                    final_atom_mask = result["structure_module"]["final_atom_mask"]
                    b_factors = result["plddt"][:, None] * final_atom_mask
                    unrelaxed_protein = protein.from_prediction(
                        features=input_features,
                        result=result, b_factors=b_factors,
                        remove_leading_feature_dimension=("multimer" not in model_type))
                    files.get("unrelaxed", f"r{recycles}.pdb").write_text(protein.to_pdb(unrelaxed_protein))

                    if save_all:
                        with files.get("all", f"r{recycles}.pickle").open("wb") as handle:
                            pickle.dump(result, handle)
                    del unrelaxed_protein

            return_representations = save_all or save_single_representations or save_pair_representations

            #zc
            # np.save(result_dir.joinpath(prefix + ".npy"), feature_dict)
            # predict
            result, recycles = \
                model_runner.predict(input_features,
                                     random_seed=seed,
                                     return_representations=return_representations,
                                     callback=callback)

            prediction_times.append(time.time() - start)

            ########################
            # parse results
            ########################
            # np.save('/home/yons/lab/zc/input_features.npy', input_features)
            # np.save('/home/yons/lab/zc/result.npy', result) #zc
            # summary metrics
            mean_scores.append(result["ranking_confidence"])
            if recycles == 0: result.pop("tol", None)
            if not is_complex: result.pop("iptm", None)
            print_line = ""
            conf.append({})
            for x, y in [["mean_plddt", "pLDDT"], ["ptm", "pTM"], ["iptm", "ipTM"]]:
                if x in result:
                    print_line += f" {y}={result[x]:.3g}"
                    conf[-1][x] = float(result[x])
            conf[-1]["print_line"] = print_line
            logger.info(f"{tag} took {prediction_times[-1]:.1f}s ({recycles} recycles)")

            # create protein object
            final_atom_mask = result["structure_module"]["final_atom_mask"]
            b_factors = result["plddt"][:, None] * final_atom_mask
            unrelaxed_protein = protein.from_prediction(
                features=input_features,
                result=result,
                b_factors=b_factors,
                remove_leading_feature_dimension=("multimer" not in model_type))

            # callback for visualization
            if prediction_callback is not None:
                prediction_callback(unrelaxed_protein, sequences_lengths,
                                    result, input_features, (tag, False))

            #########################
            # save results
            #########################      

            # save pdb
            protein_lines = protein.to_pdb(unrelaxed_protein)
            files.get("unrelaxed", "pdb").write_text(protein_lines)
            unrelaxed_pdb_lines.append(protein_lines)

            # save raw outputs
            if save_all:
                with files.get("all", "pickle").open("wb") as handle:
                    pickle.dump(result, handle)
            if save_single_representations:
                np.save(files.get("single_repr", "npy"), result["representations"]["single"])
            if save_pair_representations:
                np.save(files.get("pair_repr", "npy"), result["representations"]["pair"])

            # write an easy-to-use format (pAE and pLDDT)
            with files.get("scores", "json").open("w") as handle:
                plddt = result["plddt"][:seq_len]
                scores = {"plddt": np.around(plddt.astype(float), 2).tolist()}
                if "predicted_aligned_error" in result:
                    pae = result["predicted_aligned_error"][:seq_len, :seq_len]
                    scores.update({"max_pae": pae.max().astype(float).item(),
                                   "pae": np.around(pae.astype(float), 2).tolist()})
                    for k in ["ptm", "iptm"]:
                        if k in conf[-1]: scores[k] = np.around(conf[-1][k], 2).item()
                    del pae
                del plddt
                json.dump(scores, handle)

            del result, unrelaxed_protein

            # early stop criteria fulfilled
            if mean_scores[-1] > stop_at_score: break

        # early stop criteria fulfilled
        if mean_scores[-1] > stop_at_score: break

        # cleanup
        if "multimer" not in model_type: del input_features
    if "multimer" in model_type: del input_features

    ###################################################
    # rerank models based on predicted confidence
    ###################################################

    rank, metric = [], []
    result_files = []
    logger.info(f"reranking models by '{rank_by}' metric")
    model_rank = np.array(mean_scores).argsort()[::-1]
    for n, key in enumerate(model_rank):
        metric.append(conf[key])
        tag = model_names[key]
        files.set_tag(tag)
        # save relaxed pdb
        if n < num_relax:
            start = time.time()
            #zc
            global_indices = []
            start_index = 0
            for chain_key, positions in index_ss.items():
                chain_num = int(chain_key[1:])
                start_index = sum(feature_dict['chain_length'][:chain_num])
                global_indices.append([start_index + pos for pos in positions[0]])
            aa_names = [residue_constants.resnames[index] for index in feature_dict['aatype']]

            cyclic_peptide_indices = []
            start_index = 0
            for i, is_cyclic in enumerate(flag_nc):
                if is_cyclic == 1:
                    head_index = start_index
                    tail_index = start_index + feature_dict['chain_length'][i] - 1
                    cyclic_peptide_indices.append((head_index, tail_index))
                start_index += feature_dict['chain_length'][i]
            #zc
            pdb_lines = relax_me(dist_cst_ss=dist_cst_ss, pdb_lines=unrelaxed_pdb_lines[key], use_gpu=use_gpu_relax, disulf=global_indices, aa_names=aa_names, nc=cyclic_peptide_indices, cp=cp)
            files.get("relaxed", "pdb").write_text(pdb_lines)
            logger.info(f"Relaxation took {(time.time() - start):.1f}s")

            # zc
            os.remove("tmp.pdb")
            os.remove("leap.in")
            os.remove("leap.log")
            os.remove("system.inpcrd")
            os.remove("system.prmtop")
            # zc

        # rename files to include rank
        new_tag = f"rank_{(n + 1):03d}_{tag}"
        rank.append(new_tag)
        logger.info(f"{new_tag}{metric[-1]['print_line']}")
        for x, ext, file in files.files[tag]:
            new_file = result_dir.joinpath(f"{prefix}_{x}_{new_tag}.{ext}")
            file.rename(new_file)
            result_files.append(new_file)

    return {"rank": rank,
            "metric": metric,
            "result_files": result_files}


def parse_fasta(fasta_string: str) -> Tuple[List[str], List[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


def get_queries(
        input_path: Union[str, Path], sort_queries_by: str = "length"
) -> Tuple[List[Tuple[str, str, Optional[List[str]]]], bool]:
    """Reads a directory of fasta files, a single fasta file or a csv file and returns a tuple
    of job name, sequence and the optional a3m lines"""

    input_path = Path(input_path)
    if not input_path.exists():
        raise OSError(f"{input_path} could not be found")

    if input_path.is_file():
        if input_path.suffix == ".csv" or input_path.suffix == ".tsv":
            sep = "\t" if input_path.suffix == ".tsv" else ","
            df = pandas.read_csv(input_path, sep=sep)
            assert "id" in df.columns and "sequence" in df.columns
            queries = [
                (seq_id, sequence.upper().split(":"), None)
                for seq_id, sequence in df[["id", "sequence"]].itertuples(index=False)
            ]
            for i in range(len(queries)):
                if len(queries[i][1]) == 1:
                    queries[i] = (queries[i][0], queries[i][1][0], None)
        elif input_path.suffix == ".a3m":
            (seqs, header) = parse_fasta(input_path.read_text())
            if len(seqs) == 0:
                raise ValueError(f"{input_path} is empty")
            query_sequence = seqs[0]
            # Use a list so we can easily extend this to multiple msas later
            a3m_lines = [input_path.read_text()]
            queries = [(input_path.stem, query_sequence, a3m_lines)]
        elif input_path.suffix in [".fasta", ".faa", ".fa"]:
            (sequences, headers) = parse_fasta(input_path.read_text())
            queries = []
            for sequence, header in zip(sequences, headers):
                sequence = sequence.upper()
                if sequence.count(":") == 0:
                    # Single sequence
                    queries.append((header, sequence, None))
                else:
                    # Complex mode
                    queries.append((header, sequence.upper().split(":"), None))
        else:
            raise ValueError(f"Unknown file format {input_path.suffix}")
    else:
        assert input_path.is_dir(), "Expected either an input file or a input directory"
        queries = []
        for file in sorted(input_path.iterdir()):
            if not file.is_file():
                continue
            if file.suffix.lower() not in [".a3m", ".fasta", ".faa"]:
                logger.warning(f"non-fasta/a3m file in input directory: {file}")
                continue
            (seqs, header) = parse_fasta(file.read_text())
            if len(seqs) == 0:
                logger.error(f"{file} is empty")
                continue
            query_sequence = seqs[0]
            if len(seqs) > 1 and file.suffix in [".fasta", ".faa", ".fa"]:
                logger.warning(
                    f"More than one sequence in {file}, ignoring all but the first sequence"
                )

            if file.suffix.lower() == ".a3m":
                a3m_lines = [file.read_text()]
                queries.append((file.stem, query_sequence.upper(), a3m_lines))
            else:
                if query_sequence.count(":") == 0:
                    # Single sequence
                    queries.append((file.stem, query_sequence, None))
                else:
                    # Complex mode
                    queries.append((file.stem, query_sequence.upper().split(":"), None))

    # sort by seq. len
    if sort_queries_by == "length":
        queries.sort(key=lambda t: len("".join(t[1])))

    elif sort_queries_by == "random":
        random.shuffle(queries)

    is_complex = False
    for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(queries):
        if isinstance(query_sequence, list):
            is_complex = True
            break
        if a3m_lines is not None and a3m_lines[0].startswith("#"):
            a3m_line = a3m_lines[0].splitlines()[0]
            tab_sep_entries = a3m_line[1:].split("\t")
            if len(tab_sep_entries) == 2:
                query_seq_len = tab_sep_entries[0].split(",")
                query_seq_len = list(map(int, query_seq_len))
                query_seqs_cardinality = tab_sep_entries[1].split(",")
                query_seqs_cardinality = list(map(int, query_seqs_cardinality))
                is_single_protein = (
                    True
                    if len(query_seq_len) == 1 and query_seqs_cardinality[0] == 1
                    else False
                )
                if not is_single_protein:
                    is_complex = True
                    break
    return queries, is_complex


def pair_sequences(
        a3m_lines: List[str], query_sequences: List[str], query_cardinality: List[int]
) -> str:
    a3m_line_paired = [""] * len(a3m_lines[0].splitlines())
    for n, seq in enumerate(query_sequences):
        lines = a3m_lines[n].splitlines()
        for i, line in enumerate(lines):
            if line.startswith(">"):
                if n != 0:
                    line = line.replace(">", "\t", 1)
                a3m_line_paired[i] = a3m_line_paired[i] + line
            else:
                a3m_line_paired[i] = a3m_line_paired[i] + line * query_cardinality[n]
    return "\n".join(a3m_line_paired)


def pad_sequences(
        a3m_lines: List[str], query_sequences: List[str], query_cardinality: List[int]
) -> str:
    _blank_seq = [
        ("-" * len(seq))
        for n, seq in enumerate(query_sequences)
        for _ in range(query_cardinality[n])
    ]
    a3m_lines_combined = []
    pos = 0
    for n, seq in enumerate(query_sequences):
        for j in range(0, query_cardinality[n]):
            lines = a3m_lines[n].split("\n")
            for a3m_line in lines:
                if len(a3m_line) == 0:
                    continue
                if a3m_line.startswith(">"):
                    a3m_lines_combined.append(a3m_line)
                else:
                    a3m_lines_combined.append(
                        "".join(_blank_seq[:pos] + [a3m_line] + _blank_seq[pos + 1:])
                    )
            pos += 1
    return "\n".join(a3m_lines_combined)


def get_msa_and_templates(
        jobname: str,
        query_sequences: Union[str, List[str]],
        result_dir: Path,
        msa_mode: str,
        use_templates: bool,
        custom_template_path: str,
        pair_mode: str,
        host_url: str = DEFAULT_API_SERVER,
) -> Tuple[
    Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]
]:
    from colabfold.colabfold import run_mmseqs2

    use_env = msa_mode == "mmseqs2_uniref_env"
    if isinstance(query_sequences, str): query_sequences = [query_sequences]

    # remove duplicates before searching
    query_seqs_unique = []
    for x in query_sequences:
        if x not in query_seqs_unique:
            query_seqs_unique.append(x)

    # determine how many times is each sequence is used
    query_seqs_cardinality = [0] * len(query_seqs_unique)
    for seq in query_sequences:
        seq_idx = query_seqs_unique.index(seq)
        query_seqs_cardinality[seq_idx] += 1

    # get template features
    template_features = []
    if use_templates:
        a3m_lines_mmseqs2, template_paths = run_mmseqs2(
            query_seqs_unique,
            str(result_dir.joinpath(jobname)),
            use_env,
            use_templates=True,
            host_url=host_url,
        )
        if custom_template_path is not None:
            template_paths = {}
            for index in range(0, len(query_seqs_unique)):
                template_paths[index] = custom_template_path
        if template_paths is None:
            logger.info("No template detected")
            for index in range(0, len(query_seqs_unique)):
                template_feature = mk_mock_template(query_seqs_unique[index])
                template_features.append(template_feature)
        else:
            for index in range(0, len(query_seqs_unique)):
                if template_paths[index] is not None:
                    template_feature = mk_template(
                        a3m_lines_mmseqs2[index],
                        template_paths[index],
                        query_seqs_unique[index],
                    )
                    if len(template_feature["template_domain_names"]) == 0:
                        template_feature = mk_mock_template(query_seqs_unique[index])
                        logger.info(f"Sequence {index} found no templates")
                    else:
                        logger.info(
                            f"Sequence {index} found templates: {template_feature['template_domain_names'].astype(str).tolist()}"
                        )
                else:
                    template_feature = mk_mock_template(query_seqs_unique[index])
                    logger.info(f"Sequence {index} found no templates")

                template_features.append(template_feature)
    else:
        for index in range(0, len(query_seqs_unique)):
            template_feature = mk_mock_template(query_seqs_unique[index])
            template_features.append(template_feature)

    if len(query_sequences) == 1:
        pair_mode = "none"

    if pair_mode == "none" or pair_mode == "unpaired" or pair_mode == "unpaired_paired":
        if msa_mode == "single_sequence":
            a3m_lines = []
            num = 101
            for i, seq in enumerate(query_seqs_unique):
                a3m_lines.append(f">{num + i}\n{seq}")
        else:
            # find normal a3ms
            a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_pairing=False,
                host_url=host_url,
            )
    else:
        a3m_lines = None

    if msa_mode != "single_sequence" and (
            pair_mode == "paired" or pair_mode == "unpaired_paired"
    ):
        # find paired a3m if not a homooligomers
        if len(query_seqs_unique) > 1:
            paired_a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_pairing=True,
                host_url=host_url,
            )
        else:
            # homooligomers
            num = 101
            paired_a3m_lines = []
            for i in range(0, query_seqs_cardinality[0]):
                paired_a3m_lines.append(f">{num + i}\n{query_seqs_unique[0]}\n")
    else:
        paired_a3m_lines = None

    return (
        a3m_lines,
        paired_a3m_lines,
        query_seqs_unique,
        query_seqs_cardinality,
        template_features,
    )


def build_monomer_feature(
        sequence: str, unpaired_msa: str, template_features: Dict[str, Any]
):
    msa = pipeline.parsers.parse_a3m(unpaired_msa)
    # gather features
    return {
        **pipeline.make_sequence_features(
            sequence=sequence, description="none", num_res=len(sequence)
        ),
        **pipeline.make_msa_features([msa]),
        **template_features,
    }


def build_multimer_feature(paired_msa: str) -> Dict[str, ndarray]:
    parsed_paired_msa = pipeline.parsers.parse_a3m(paired_msa)
    return {
        f"{k}_all_seq": v
        for k, v in pipeline.make_msa_features([parsed_paired_msa]).items()
    }


def process_multimer_features(
        disulfide_bond_pairs: List[Tuple[int]],
        flag_cyclic_peptide: List[int],
        flag_nc: List[int],
        index_ss: Dict[List[Tuple[int]]],
        features_for_chain: Dict[str, Dict[str, ndarray]],
        min_num_seq: int = 512,
) -> Dict[str, ndarray]:
    all_chain_features = {}
    for chain_id, chain_features in features_for_chain.items():
        all_chain_features[chain_id] = pipeline_multimer.convert_monomer_features(
            chain_features, chain_id
        )

    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    # np_example = feature_processing.pair_and_merge(
    #    all_chain_features=all_chain_features, is_prokaryote=is_prokaryote)
    feature_processing.process_unmerged_features(all_chain_features)
    np_chains_list = list(all_chain_features.values())
    # noinspection PyProtectedMember
    pair_msa_sequences = not feature_processing._is_homomer_or_monomer(np_chains_list)
    chains = list(np_chains_list)
    chain_keys = chains[0].keys()
    updated_chains = []
    for chain_num, chain in enumerate(chains):
        new_chain = {k: v for k, v in chain.items() if "_all_seq" not in k}
        for feature_name in chain_keys:
            if feature_name.endswith("_all_seq"):
                feats_padded = msa_pairing.pad_features(
                    chain[feature_name], feature_name
                )
                new_chain[feature_name] = feats_padded
        new_chain["num_alignments_all_seq"] = np.asarray(
            len(np_chains_list[chain_num]["msa_all_seq"])
        )
        updated_chains.append(new_chain)
    np_chains_list = updated_chains
    np_chains_list = feature_processing.crop_chains(
        np_chains_list,
        msa_crop_size=feature_processing.MSA_CROP_SIZE,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=feature_processing.MAX_TEMPLATES,
    )
    # merge_chain_features crashes if there are additional features only present in one chain
    # remove all features that are not present in all chains
    common_features = set([*np_chains_list[0]]).intersection(*np_chains_list)
    np_chains_list = [
        {key: value for (key, value) in chain.items() if key in common_features}
        for chain in np_chains_list
    ]
    np_example = feature_processing.msa_pairing.merge_chain_features(
        np_chains_list=np_chains_list,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=feature_processing.MAX_TEMPLATES,
    )
    np_example = feature_processing.process_final(np_example)

    # Pad MSA to avoid zero-sized extra_msa.
    np_example = pipeline_multimer.pad_msa(np_example, min_num_seq=min_num_seq)

    # start of add on 20230517
    # get_offset(disulfide_bond_pairs, np_example)
    get_offset_multicyc(disulfide_bond_pairs, flag_cyclic_peptide, flag_nc, index_ss, np_example)
    # end of add on 20230517

    return np_example


def get_offset_multicyc(
        disulfide_bond_pairs: List[Tuple[int]],
        flag_cyclic_peptide: List[int],
        flag_nc: List[int],
        index_ss: Dict[List[Tuple[int]]],
        feat: Dict[str, ndarray]
):
    # print(feat['residue_index'])
    # print(flag_cyclic_peptide)
    # print(flag_nc)
    # print(index_ss)
    feat['flag_cyclic_peptide'] = np.array(flag_cyclic_peptide)
    index_zero = [i for i in range(len(feat['residue_index'])) if feat['residue_index'][i] == 0]
    n_chain = len(index_zero)
    
    if not flag_nc:
        flag_nc = [0]*n_chain
    if not flag_cyclic_peptide:
        flag_cyclic_peptide = [0]*n_chain

    feat['cycpep_offset'] = []  # 位置编码矩阵
    feat['chain_start_pos'] = []  # 每条序列的起始位置
    feat['chain_length'] = []  # 每条序列的长度
    for i in range(n_chain):
        feat['cycpep_offset'].append(np.array([]).astype(int))
        feat['chain_start_pos'].append(np.array(index_zero[i]))
        if i < n_chain - 1:
            feat['chain_length'].append(np.array(index_zero[i+1]-index_zero[i]))
        else:
            feat['chain_length'].append(np.array(len(feat['residue_index'])-index_zero[i]))

#    if flag_cyclic_peptide and (flag_nc or index_ss):  
    for i in range(n_chain):
        if flag_cyclic_peptide[i] == 1:               # 存在环肽序列，需要构造位置编码矩阵
            c_start = []  # c_start[i]和c_end[i]成环
            c_end = []
            if index_ss.get('c'+str(i)):
                for item in index_ss.get('c'+str(i)):
                    c_start.append(item[0])
                    c_end.append(item[1])
            feat['cycpep_offset'][i] = np.array(calc_offset_matrix(feat['chain_length'][i], c_start, c_end, flag_nc[i])).astype(int)



def get_offset(
        disulfide_bond_pairs: List[Tuple[int]],
        feat: Dict[str, ndarray]
):
    feat['cycpep_index'] = np.arange(1 + feat['residue_index'][-1])
    #
    # [0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    #  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
    #  48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64  0  1  2  3  4  5  6
    #  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25  0  1  2  3  4
    #  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25]
    len_cycpep = len(feat['cycpep_index'])
    c_start = []  # c_start[i]和c_end[i]成环
    c_end = []
    for item in disulfide_bond_pairs:
        c_start.append(item[0])
        c_end.append(item[1])
    feat['offset_ss'] = np.array(calc_offset_matrix(len_cycpep, c_start, c_end)).astype(int)
    # [[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9. 8.  7.  6.  5.  4.  3.  2.  1.]
    #  [ 1.  0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10. 9.  8.  7.  6.  5.  4.  3.  2.]
    #  [ 2.  1.  0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.  6.  5.  4.  3.]
    #  [ 3.  2.  1.  0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.  6.  5.  4.]
    #  [ 4.  3.  2.  1.  0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.  6.  5.]
    #  [ 5.  4.  3.  2.  1.  0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.  6.]
    #  [ 6.  5.  4.  3.  2.  1.  0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.]
    #  [ 7.  6.  5.  4.  3.  2.  1.  0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.]
    #  [ 8.  7.  6.  5.  4.  3.  2.  1.  0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.]
    #  [ 9.  8.  7.  6.  5.  4.  3.  2.  1.  0.  1.  2.  3.  4.  5.  6.  7.  8. 9. 10. 11. 12. 13. 12. 11. 10.]
    #  [10.  9.  8.  7.  6.  5.  4.  3.  2.  1.  0.  1.  2.  3.  4.  5.  6.  7. 8.  9. 10. 11. 12. 13. 12. 11.]
    #  [11. 10.  9.  8.  7.  6.  5.  4.  3.  2.  1.  0.  1.  2.  3.  4.  5.  6.
    #    7.  8.  9. 10. 11. 12. 13. 12.]
    #  [12. 11. 10.  9.  8.  7.  6.  5.  4.  3.  2.  1.  0.  1.  2.  3.  4.  5.
    #    6.  7.  8.  9. 10. 11. 12. 13.]
    #  [13. 12. 11. 10.  9.  8.  7.  6.  5.  4.  3.  2.  1.  0.  1.  2.  3.  4.
    #    5.  6.  7.  8.  9. 10. 11. 12.]
    #  [12. 13. 12. 11. 10.  9.  8.  7.  6.  5.  4.  3.  2.  1.  0.  1.  2.  3.
    #    4.  5.  6.  7.  8.  9. 10. 11.]
    #  [11. 12. 13. 12. 11. 10.  9.  8.  7.  6.  5.  4.  3.  2.  1.  0.  1.  2.
    #    3.  4.  5.  6.  7.  8.  9. 10.]
    #  [10. 11. 12. 13. 12. 11. 10.  9.  8.  7.  6.  5.  4.  3.  2.  1.  0.  1.
    #    2.  3.  4.  5.  6.  7.  8.  9.]
    #  [ 9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.  6.  5.  4.  3.  2.  1.  0.
    #    1.  2.  3.  4.  5.  6.  7.  8.]
    #  [ 8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.  6.  5.  4.  3.  2.  1.
    #    0.  1.  2.  3.  4.  5.  6.  7.]
    #  [ 7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.  6.  5.  4.  3.  2.
    #    1.  0.  1.  2.  3.  4.  5.  6.]
    #  [ 6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.  6.  5.  4.  3.
    #    2.  1.  0.  1.  2.  3.  4.  5.]
    #  [ 5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.  6.  5.  4.
    #    3.  2.  1.  0.  1.  2.  3.  4.]
    #  [ 4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.  6.  5.
    #    4.  3.  2.  1.  0.  1.  2.  3.]
    #  [ 3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.  6.
    #    5.  4.  3.  2.  1.  0.  1.  2.]
    #  [ 2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.  7.
    #    6.  5.  4.  3.  2.  1.  0.  1.]
    #  [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 12. 11. 10.  9.  8.
    #    7.  6.  5.  4.  3.  2.  1.  0.]]


def calc_offset_matrix(n_aa, c1, c2, flag_nc=1):
    """
    :param n_aa: number of amino acid residues
    :param c1: list of Cys residues
    :param c2: list of the corresponding Cys residues to c1, the same length with c1
    :return:
    """
    if len(c1) != len(c2):
        return []

    # init adjacency matrix
    matrix = np.zeros((n_aa, n_aa)) + n_aa
    for i in range(n_aa):
        matrix[i][i] = 0

    # linear peptide connection
    for i in range(n_aa - 1):
        matrix[i][i + 1] = 1
        matrix[i + 1][i] = 1

    # nc connection
    if flag_nc:
        matrix[0][n_aa - 1] = 1
        matrix[n_aa - 1][0] = 1

    # ss connection
    for i in range(len(c1)):
        matrix[c1[i]][c2[i]] = 1
        matrix[c2[i]][c1[i]] = 1

    # get the shortest path
    matrix = get_opt_path(matrix)
    
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[0], 1):
            matrix[i][j] *= -1

    return matrix


def get_opt_path(matrix):
    """
    Floyd algorithm to find the shortest path
    :param matrix:
    :return:
    """
    path = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        path[i] = [j for j in range(matrix.shape[0])]
    # print(path)
    # print()
    # print(matrix)

    for m in range(matrix.shape[0]):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                if matrix[i][m] + matrix[m][j] < matrix[i][j]:
                    matrix[i][j] = matrix[i][m] + matrix[m][j]
                    path[i][j] = m
    # print()
    # print(path)
    # print()
    # print(matrix)

    return matrix


def pair_msa(
        query_seqs_unique: List[str],
        query_seqs_cardinality: List[int],
        paired_msa: Optional[List[str]],
        unpaired_msa: Optional[List[str]],
) -> str:
    if paired_msa is None and unpaired_msa is not None:
        a3m_lines = pad_sequences(
            unpaired_msa, query_seqs_unique, query_seqs_cardinality
        )
    elif paired_msa is not None and unpaired_msa is not None:
        a3m_lines = (
                pair_sequences(paired_msa, query_seqs_unique, query_seqs_cardinality)
                + "\n"
                + pad_sequences(unpaired_msa, query_seqs_unique, query_seqs_cardinality)
        )
    elif paired_msa is not None and unpaired_msa is None:
        a3m_lines = pair_sequences(
            paired_msa, query_seqs_unique, query_seqs_cardinality
        )
    else:
        raise ValueError(f"Invalid pairing")
    return a3m_lines


def generate_input_feature(
        disulfide_bond_pairs: List[Tuple[int]],
        flag_cyclic_peptide: List[int],
        flag_nc: List[int],
        index_ss: Dict[List[Tuple[int]]],
        query_seqs_unique: List[str],
        query_seqs_cardinality: List[int],
        unpaired_msa: List[str],
        paired_msa: List[str],
        template_features: List[Dict[str, Any]],
        is_complex: bool,
        model_type: str,
        max_seq: int,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    input_feature = {}
    domain_names = {}
    if is_complex and "multimer" not in model_type:

        full_sequence = ""
        Ls = []
        for sequence_index, sequence in enumerate(query_seqs_unique):
            for cardinality in range(0, query_seqs_cardinality[sequence_index]):
                full_sequence += sequence
                Ls.append(len(sequence))

        # bugfix
        a3m_lines = f">0\n{full_sequence}\n"
        a3m_lines += pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)

        input_feature = build_monomer_feature(full_sequence, a3m_lines, mk_mock_template(full_sequence))
        input_feature["residue_index"] = np.concatenate([np.arange(L) for L in Ls])
        input_feature["asym_id"] = np.concatenate([np.full(L, n) for n, L in enumerate(Ls)])
        if any(
                [
                    template != b"none"
                    for i in template_features
                    for template in i["template_domain_names"]
                ]
        ):
            logger.warning(
                "alphafold2_ptm complex does not consider templates. Chose multimer model-type for template support."
            )

    else:
        features_for_chain = {}
        chain_cnt = 0
        # for each unique sequence
        for sequence_index, sequence in enumerate(query_seqs_unique):

            # get unpaired msa
            if unpaired_msa is None:
                input_msa = f">{101 + sequence_index}\n{sequence}"
            else:
                input_msa = unpaired_msa[sequence_index]

            feature_dict = build_monomer_feature(
                sequence, input_msa, template_features[sequence_index])

            if "multimer" in model_type:
                # get paired msa
                if paired_msa is None:
                    input_msa = f">{101 + sequence_index}\n{sequence}"
                else:
                    input_msa = paired_msa[sequence_index]
                feature_dict.update(build_multimer_feature(input_msa))

            # for each copy
            for cardinality in range(0, query_seqs_cardinality[sequence_index]):
                features_for_chain[protein.PDB_CHAIN_IDS[chain_cnt]] = feature_dict
                chain_cnt += 1

        if "multimer" in model_type:
            # combine features across all chains
            input_feature = process_multimer_features(disulfide_bond_pairs, flag_cyclic_peptide, flag_nc, index_ss,
                                                      features_for_chain, min_num_seq=max_seq + 4)
            domain_names = {
                chain: [
                    name.decode("UTF-8")
                    for name in feature["template_domain_names"]
                    if name != b"none"
                ]
                for (chain, feature) in features_for_chain.items()
            }
        else:
            input_feature = features_for_chain[protein.PDB_CHAIN_IDS[0]]
            input_feature["asym_id"] = np.zeros(input_feature["aatype"].shape[0], dtype=int)
            domain_names = {
                protein.PDB_CHAIN_IDS[0]: [
                    name.decode("UTF-8")
                    for name in input_feature["template_domain_names"]
                    if name != b"none"
                ]
            }
    return (input_feature, domain_names)


def unserialize_msa(
        a3m_lines: List[str], query_sequence: Union[List[str], str]
) -> Tuple[
    Optional[List[str]],
    Optional[List[str]],
    List[str],
    List[int],
    List[Dict[str, Any]],
]:
    a3m_lines = a3m_lines[0].replace("\x00", "").splitlines()
    if not a3m_lines[0].startswith("#") or len(a3m_lines[0][1:].split("\t")) != 2:
        assert isinstance(query_sequence, str)
        return (
            ["\n".join(a3m_lines)],
            None,
            [query_sequence],
            [1],
            [mk_mock_template(query_sequence)],
        )

    if len(a3m_lines) < 3:
        raise ValueError(f"Unknown file format a3m")
    tab_sep_entries = a3m_lines[0][1:].split("\t")
    query_seq_len = tab_sep_entries[0].split(",")
    query_seq_len = list(map(int, query_seq_len))
    query_seqs_cardinality = tab_sep_entries[1].split(",")
    query_seqs_cardinality = list(map(int, query_seqs_cardinality))
    is_homooligomer = (
        True if len(query_seq_len) == 1 and query_seqs_cardinality[0] > 1 else False
    )
    is_single_protein = (
        True if len(query_seq_len) == 1 and query_seqs_cardinality[0] == 1 else False
    )
    query_seqs_unique = []
    prev_query_start = 0
    # we store the a3m with cardinality of 1
    for n, query_len in enumerate(query_seq_len):
        query_seqs_unique.append(
            a3m_lines[2][prev_query_start: prev_query_start + query_len]
        )
        prev_query_start += query_len
    paired_msa = [""] * len(query_seq_len)
    unpaired_msa = [""] * len(query_seq_len)
    already_in = dict()
    for i in range(1, len(a3m_lines), 2):
        header = a3m_lines[i]
        seq = a3m_lines[i + 1]
        if (header, seq) in already_in:
            continue
        already_in[(header, seq)] = 1
        has_amino_acid = [False] * len(query_seq_len)
        seqs_line = []
        prev_pos = 0
        for n, query_len in enumerate(query_seq_len):
            paired_seq = ""
            curr_seq_len = 0
            for pos in range(prev_pos, len(seq)):
                if curr_seq_len == query_len:
                    prev_pos = pos
                    break
                paired_seq += seq[pos]
                if seq[pos].islower():
                    continue
                if seq[pos] != "-":
                    has_amino_acid[n] = True
                curr_seq_len += 1
            seqs_line.append(paired_seq)

        # is sequence is paired add them to output
        if (
                not is_single_protein
                and not is_homooligomer
                and sum(has_amino_acid) == len(query_seq_len)
        ):
            header_no_faster = header.replace(">", "")
            header_no_faster_split = header_no_faster.split("\t")
            for j in range(0, len(seqs_line)):
                paired_msa[j] += ">" + header_no_faster_split[j] + "\n"
                paired_msa[j] += seqs_line[j] + "\n"
        else:
            for j, seq in enumerate(seqs_line):
                if has_amino_acid[j]:
                    unpaired_msa[j] += header + "\n"
                    unpaired_msa[j] += seq + "\n"
    if is_homooligomer:
        # homooligomers
        num = 101
        paired_msa = [""] * query_seqs_cardinality[0]
        for i in range(0, query_seqs_cardinality[0]):
            paired_msa[i] = ">" + str(num + i) + "\n" + query_seqs_unique[0] + "\n"
    if is_single_protein:
        paired_msa = None
    template_features = []
    for query_seq in query_seqs_unique:
        template_feature = mk_mock_template(query_seq)
        template_features.append(template_feature)

    return (
        unpaired_msa,
        paired_msa,
        query_seqs_unique,
        query_seqs_cardinality,
        template_features,
    )


def msa_to_str(
        unpaired_msa: List[str],
        paired_msa: List[str],
        query_seqs_unique: List[str],
        query_seqs_cardinality: List[int],
) -> str:
    msa = "#" + ",".join(map(str, map(len, query_seqs_unique))) + "\t"
    msa += ",".join(map(str, query_seqs_cardinality)) + "\n"
    # build msa with cardinality of 1, it makes it easier to parse and manipulate
    query_seqs_cardinality = [1 for _ in query_seqs_cardinality]
    msa += pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)
    return msa


def run(
        unnatural_residue: List[str],
        disulfide_bond_pairs: List[Tuple[int]],
        flag_cyclic_peptide: List[int],
        flag_nc: List[int],
        index_ss: Dict[List[Tuple[int]]],
        dist_cst_ss: List[List[int]],
        queries: List[Tuple[str, Union[str, List[str]], Optional[List[str]]]],
        result_dir: Union[str, Path],
        num_models: int,
        is_complex: bool,
        num_recycles: Optional[int] = None,
        recycle_early_stop_tolerance: Optional[float] = None,
        model_order: List[int] = [1, 2, 3, 4, 5],
        num_ensemble: int = 1,
        model_type: str = "auto",
        msa_mode: str = "mmseqs2_uniref_env",
        use_templates: bool = False,
        custom_template_path: str = None,
        num_relax: int = 0,
        keep_existing_results: bool = True,
        rank_by: str = "auto",
        pair_mode: str = "unpaired_paired",
        data_dir: Union[str, Path] = default_data_dir,
        host_url: str = DEFAULT_API_SERVER,
        random_seed: int = 0,
        num_seeds: int = 1,
        recompile_padding: Union[int, float] = 10,
        zip_results: bool = False,
        prediction_callback: Callable[[Any, Any, Any, Any, Any], Any] = None,
        save_single_representations: bool = False,
        save_pair_representations: bool = False,
        save_all: bool = False,
        save_recycles: bool = False,
        use_dropout: bool = False,
        use_gpu_relax: bool = False,
        stop_at_score: float = 100,
        dpi: int = 200,
        max_seq: Optional[int] = None,
        max_extra_seq: Optional[int] = None,
        use_cluster_profile: bool = True,
        feature_dict_callback: Callable[[Any], Any] = None,
        **kwargs
):
    # check what device is available
    try:
        # check if TPU is available
        import jax.tools.colab_tpu
        jax.tools.colab_tpu.setup_tpu()
        logger.info('Running on TPU')
        DEVICE = "tpu"
        use_gpu_relax = False
    except:
        if jax.local_devices()[0].platform == 'cpu':
            logger.info("WARNING: no GPU detected, will be using CPU")
            DEVICE = "cpu"
            use_gpu_relax = False
        else:
            import tensorflow as tf
            tf.get_logger().setLevel(logging.ERROR)
            logger.info('Running on GPU')
            DEVICE = "gpu"
            # disable GPU on tensorflow
            tf.config.set_visible_devices([], 'GPU')

    from alphafold.notebooks.notebook_utils import get_pae_json
    from colabfold.alphafold.models import load_models_and_params
    from colabfold.colabfold import plot_paes, plot_plddts
    from colabfold.plot import plot_msa_v2

    data_dir = Path(data_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    model_type = set_model_type(is_complex, model_type)

    # determine model extension
    if model_type == "alphafold2_multimer_v1":
        model_suffix = "_multimer"
    elif model_type == "alphafold2_multimer_v2":
        model_suffix = "_multimer_v2"
    elif model_type == "alphafold2_multimer_v3":
        model_suffix = "_multimer_v3"
    elif model_type == "alphafold2_ptm":
        model_suffix = "_ptm"
    elif model_type == "alphafold2":
        model_suffix = ""
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    # backward-compatibility with old options
    old_names = {"MMseqs2 (UniRef+Environmental)": "mmseqs2_uniref_env",
                 "MMseqs2 (UniRef only)": "mmseqs2_uniref",
                 "unpaired+paired": "unpaired_paired"}
    msa_mode = old_names.get(msa_mode, msa_mode)
    pair_mode = old_names.get(pair_mode, pair_mode)
    feature_dict_callback = kwargs.pop("input_features_callback", feature_dict_callback)
    use_dropout = kwargs.pop("training", use_dropout)
    use_fuse = kwargs.pop("use_fuse", True)
    use_bfloat16 = kwargs.pop("use_bfloat16", True)
    max_msa = kwargs.pop("max_msa", None)
    if max_msa is not None:
        max_seq, max_extra_seq = [int(x) for x in max_msa.split(":")]

    if kwargs.pop("use_amber", False) and num_relax == 0:
        num_relax = num_models * num_seeds

    if len(kwargs) > 0:
        print(f"WARNING: the following options are not being used: {kwargs}")

    # decide how to rank outputs
    if rank_by == "auto":
        rank_by = "multimer" if is_complex else "plddt"
    if "ptm" not in model_type and "multimer" not in model_type:
        rank_by = "plddt"

    # get max length
    max_len = 0
    max_num = 0
    for _, query_sequence, _ in queries:
        N = 1 if isinstance(query_sequence, str) else len(query_sequence)
        L = len("".join(query_sequence))
        if L > max_len: max_len = L
        if N > max_num: max_num = N

    # get max sequences
    # 512 5120 = alphafold_ptm (models 1,3,4)
    # 512 1024 = alphafold_ptm (models 2,5)
    # 508 2048 = alphafold-multimer_v3 (models 1,2,3)
    # 508 1152 = alphafold-multimer_v3 (models 4,5)
    # 252 1152 = alphafold-multimer_v[1,2]
    set_if = lambda x, y: y if x is None else x
    if model_type in ["alphafold2_multimer_v1", "alphafold2_multimer_v2"]:
        (max_seq, max_extra_seq) = (set_if(max_seq, 252), set_if(max_extra_seq, 1152))
    elif model_type == "alphafold2_multimer_v3":
        (max_seq, max_extra_seq) = (set_if(max_seq, 508), set_if(max_extra_seq, 2048))
    else:
        (max_seq, max_extra_seq) = (set_if(max_seq, 512), set_if(max_extra_seq, 5120))

    if msa_mode == "single_sequence":
        num_seqs = 1
        if is_complex and "multimer" not in model_type: num_seqs += max_num
        if use_templates: num_seqs += 4
        max_seq = min(num_seqs, max_seq)
        max_extra_seq = max(min(num_seqs - max_seq, max_extra_seq), 1)

    # sort model order
    model_order.sort()

    # Record the parameters of this run
    config = {
        "num_queries": len(queries),
        "use_templates": use_templates,
        "num_relax": num_relax,
        "msa_mode": msa_mode,
        "model_type": model_type,
        "num_models": num_models,
        "num_recycles": num_recycles,
        "recycle_early_stop_tolerance": recycle_early_stop_tolerance,
        "num_ensemble": num_ensemble,
        "model_order": model_order,
        "keep_existing_results": keep_existing_results,
        "rank_by": rank_by,
        "max_seq": max_seq,
        "max_extra_seq": max_extra_seq,
        "pair_mode": pair_mode,
        "host_url": host_url,
        "stop_at_score": stop_at_score,
        "random_seed": random_seed,
        "num_seeds": num_seeds,
        "recompile_padding": recompile_padding,
        "commit": get_commit(),
        "use_dropout": use_dropout,
        "use_cluster_profile": use_cluster_profile,
        "use_fuse": use_fuse,
        "use_bfloat16": use_bfloat16,
        "version": importlib_metadata.version("colabfold"),
    }
    config_out_file = result_dir.joinpath("config.json")
    config_out_file.write_text(json.dumps(config, indent=4))
    use_env = "env" in msa_mode
    use_msa = "mmseqs2" in msa_mode
    use_amber = num_relax > 0

    bibtex_file = write_bibtex(
        model_type, use_msa, use_env, use_templates, use_amber, result_dir
    )

    if custom_template_path is not None:
        mk_hhsearch_db(custom_template_path)

    pad_len = 0
    ranks, metrics = [], []
    first_job = True
    for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(queries):
        jobname = safe_filename(raw_jobname)

        #######################################
        # check if job has already finished
        #######################################
        # In the colab version and with --zip we know we're done when a zip file has been written
        result_zip = result_dir.joinpath(jobname).with_suffix(".result.zip")
        if keep_existing_results and result_zip.is_file():
            logger.info(f"Skipping {jobname} (result.zip)")
            continue
        # In the local version we use a marker file
        is_done_marker = result_dir.joinpath(jobname + ".done.txt")
        if keep_existing_results and is_done_marker.is_file():
            logger.info(f"Skipping {jobname} (already done)")
            continue

        seq_len = len("".join(query_sequence))
        logger.info(f"Query {job_number + 1}/{len(queries)}: {jobname} (length {seq_len})")

        ###########################################
        # generate MSA (a3m_lines) and templates
        ###########################################
        try:
            if use_templates or a3m_lines is None:
                (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) \
                    = get_msa_and_templates(jobname, query_sequence, result_dir, msa_mode, use_templates,
                                            custom_template_path, pair_mode, host_url)
            if a3m_lines is not None:
                (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features_) \
                    = unserialize_msa(a3m_lines, query_sequence)
                if not use_templates: template_features = template_features_

            # save a3m
            msa = msa_to_str(unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality)
            result_dir.joinpath(f"{jobname}.a3m").write_text(msa)

        except Exception as e:
            logger.exception(f"Could not get MSA/templates for {jobname}: {e}")
            continue

        #######################
        # generate features
        #######################
        try:
            (feature_dict, domain_names) \
                = generate_input_feature(disulfide_bond_pairs, flag_cyclic_peptide, flag_nc, index_ss,
                                         query_seqs_unique, query_seqs_cardinality, unpaired_msa, paired_msa,
                                         template_features, is_complex, model_type, max_seq=max_seq)

            # to allow display of MSA info during colab/chimera run (thanks tomgoddard)
            if feature_dict_callback is not None:
                feature_dict_callback(feature_dict)

            # np.save(result_dir.joinpath(jobname + ".npy"), feature_dict)

        except Exception as e:
            logger.exception(f"Could not generate input features {jobname}: {e}")
            continue

        
        # zc
        unnatural = unnatural_residue
        indices = np.where(feature_dict['aatype'] == len(residue_constants.restypes))[0]
        if model_type == 'alphafold2_multimer_v3':
            assert len(indices) == len(unnatural)
            for i, name in enumerate(unnatural):
                feature_dict['aatype'][indices[i]] = residue_constants.restypes.index(name)

        # np.save(result_dir.joinpath(jobname + ".npy"), feature_dict)

        ######################
        # predict structures
        ######################
        try:
            # get list of lengths
            query_sequence_len_array = sum([[len(x)] * y
                                            for x, y in zip(query_seqs_unique, query_seqs_cardinality)], [])

            # decide how much to pad (to avoid recompiling)
            if seq_len > pad_len:
                if isinstance(recompile_padding, float):
                    pad_len = math.ceil(seq_len * recompile_padding)
                else:
                    pad_len = seq_len + recompile_padding
                pad_len = min(pad_len, max_len)

            # prep model and params
            if first_job:
                # if one job input adjust max settings
                if len(queries) == 1 and msa_mode != "single_sequence":
                    # get number of sequences
                    if "msa_mask" in feature_dict:
                        num_seqs = int(sum(feature_dict["msa_mask"].max(-1) == 1))
                    else:
                        num_seqs = int(len(feature_dict["msa"]))

                    if use_templates: num_seqs += 4

                    # adjust max settings
                    max_seq = min(num_seqs, max_seq)
                    max_extra_seq = max(min(num_seqs - max_seq, max_extra_seq), 1)
                    logger.info(f"Setting max_seq={max_seq}, max_extra_seq={max_extra_seq}")

                model_runner_and_params = load_models_and_params(
                    num_models=num_models,
                    use_templates=use_templates,
                    num_recycles=num_recycles,
                    num_ensemble=num_ensemble,
                    model_order=model_order,
                    model_suffix=model_suffix,
                    data_dir=data_dir,
                    stop_at_score=stop_at_score,
                    rank_by=rank_by,
                    use_dropout=use_dropout,
                    max_seq=max_seq,
                    max_extra_seq=max_extra_seq,
                    use_cluster_profile=use_cluster_profile,
                    recycle_early_stop_tolerance=recycle_early_stop_tolerance,
                    use_fuse=use_fuse,
                    use_bfloat16=use_bfloat16,
                    save_all=save_all,
                )
                first_job = False

            results = predict_structure(
                dist_cst_ss=dist_cst_ss,
                index_ss=index_ss,
                prefix=jobname,
                result_dir=result_dir,
                feature_dict=feature_dict,
                is_complex=is_complex,
                use_templates=use_templates,
                sequences_lengths=query_sequence_len_array,
                pad_len=pad_len,
                model_type=model_type,
                model_runner_and_params=model_runner_and_params,
                num_relax=num_relax,
                rank_by=rank_by,
                stop_at_score=stop_at_score,
                prediction_callback=prediction_callback,
                use_gpu_relax=use_gpu_relax,
                random_seed=random_seed,
                num_seeds=num_seeds,
                save_all=save_all,
                save_single_representations=save_single_representations,
                save_pair_representations=save_pair_representations,
                save_recycles=save_recycles,
                flag_nc=flag_nc,
                cp=flag_cyclic_peptide,
            )
            result_files = results["result_files"]
            ranks.append(results["rank"])
            metrics.append(results["metric"])


        except RuntimeError as e:
            # This normally happens on OOM. TODO: Filter for the specific OOM error message
            logger.error(f"Could not predict {jobname}. Not Enough GPU memory? {e}")
            continue

        ###############
        # save plots
        ###############

        # make msa plot
        msa_plot = plot_msa_v2(feature_dict, dpi=dpi)
        coverage_png = result_dir.joinpath(f"{jobname}_coverage.png")
        msa_plot.savefig(str(coverage_png), bbox_inches='tight')
        msa_plot.close()
        result_files.append(coverage_png)

        # load the scores
        scores = []
        for r in results["rank"][:5]:
            scores_file = result_dir.joinpath(f"{jobname}_scores_{r}.json")
            with scores_file.open("r") as handle:
                scores.append(json.load(handle))

        # write alphafold-db format (pAE)
        if "pae" in scores[0]:
            af_pae_file = result_dir.joinpath(f"{jobname}_predicted_aligned_error_v1.json")
            af_pae_file.write_text(json.dumps({
                "predicted_aligned_error": scores[0]["pae"],
                "max_predicted_aligned_error": scores[0]["max_pae"]}))
            result_files.append(af_pae_file)

            # make pAE plots
            paes_plot = plot_paes([np.asarray(x["pae"]) for x in scores],
                                  Ls=query_sequence_len_array, dpi=dpi)
            pae_png = result_dir.joinpath(f"{jobname}_pae.png")
            paes_plot.savefig(str(pae_png), bbox_inches='tight')
            paes_plot.close()
            result_files.append(pae_png)

        # make pLDDT plot
        plddt_plot = plot_plddts([np.asarray(x["plddt"]) for x in scores],
                                 Ls=query_sequence_len_array, dpi=dpi)
        plddt_png = result_dir.joinpath(f"{jobname}_plddt.png")
        plddt_plot.savefig(str(plddt_png), bbox_inches='tight')
        plddt_plot.close()
        result_files.append(plddt_png)

        if use_templates:
            templates_file = result_dir.joinpath(f"{jobname}_template_domain_names.json")
            templates_file.write_text(json.dumps(domain_names))
            result_files.append(templates_file)

        result_files.append(result_dir.joinpath(jobname + ".a3m"))
        result_files += [bibtex_file, config_out_file]

        if zip_results:
            with zipfile.ZipFile(result_zip, "w") as result_zip:
                for file in result_files:
                    result_zip.write(file, arcname=file.name)

            # Delete only after the zip was successful, and also not the bibtex and config because we need those again
            for file in result_files[:-2]:
                file.unlink()
        else:
            is_done_marker.touch()

    logger.info("Done")
    return {"rank": ranks, "metric": metrics}


def set_model_type(is_complex: bool, model_type: str) -> str:
    # backward-compatibility with old options
    old_names = {"AlphaFold2-multimer-v1": "alphafold2_multimer_v1",
                 "AlphaFold2-multimer-v2": "alphafold2_multimer_v2",
                 "AlphaFold2-multimer-v3": "alphafold2_multimer_v3",
                 "AlphaFold2-ptm": "alphafold2_ptm",
                 "AlphaFold2": "alphafold2"}
    model_type = old_names.get(model_type, model_type)
    if model_type == "auto":
        if is_complex:
            model_type = "alphafold2_multimer_v3"
        else:
            model_type = "alphafold2_ptm"
    return model_type


def main():
    parser = ArgumentParser()
    parser.add_argument("input",
                        default="input",
                        help="Can be one of the following: "
                             "Directory with fasta/a3m files, a csv/tsv file, a fasta file or an a3m file",
                        )
    parser.add_argument("results", help="Directory to write the results to")
    parser.add_argument("--stop-at-score",
                        help="Compute models until plddt (single chain) or ptmscore (complex) > threshold is reached. "
                             "This can make colabfold much faster by only running the first model for easy queries.",
                        type=float,
                        default=100,
                        )
    parser.add_argument("--num-recycle",
                        help="Number of prediction recycles."
                             "Increasing recycles can improve the quality but slows down the prediction.",
                        type=int,
                        default=None,
                        )
    parser.add_argument("--recycle-early-stop-tolerance",
                        help="Specify convergence criteria."
                             "Run until the distance between recycles is within specified value.",
                        type=float,
                        default=None,
                        )
    parser.add_argument("--num-ensemble",
                        help="Number of ensembles."
                             "The trunk of the network is run multiple times with different random choices for the MSA cluster centers.",
                        type=int,
                        default=1,
                        )
    parser.add_argument("--num-seeds",
                        help="Number of seeds to try. Will iterate from range(random_seed, random_seed+num_seeds)."
                             ".",
                        type=int,
                        default=1,
                        )
    parser.add_argument("--random-seed",
                        help="Changing the seed for the random number generator can result in different structure predictions.",
                        type=int,
                        default=0,
                        )
    parser.add_argument("--num-models", type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--recompile-padding",
                        type=int,
                        default=10,
                        help="Whenever the input length changes, the model needs to be recompiled."
                             "We pad sequences by specified length, so we can e.g. compute sequence from length 100 to 110 without recompiling."
                             "The prediction will become marginally slower for the longer input, "
                             "but overall performance increases due to not recompiling. "
                             "Set to 0 to disable.",
                        )
    parser.add_argument("--model-order", default="1,2,3,4,5", type=str)
    parser.add_argument("--host-url", default=DEFAULT_API_SERVER)
    parser.add_argument("--data")
    parser.add_argument("--msa-mode",
                        default="mmseqs2_uniref_env",
                        choices=[
                            "mmseqs2_uniref_env",
                            "mmseqs2_uniref",
                            "single_sequence",
                        ],
                        help="Using an a3m file as input overwrites this option",
                        )
    parser.add_argument("--model-type",
                        help="predict strucutre/complex using the following model."
                             'Auto will pick "alphafold2_ptm" for structure predictions and "alphafold2_multimer_v3" for complexes.',
                        type=str,
                        default="auto",
                        choices=[
                            "auto",
                            "alphafold2",
                            "alphafold2_ptm",
                            "alphafold2_multimer_v1",
                            "alphafold2_multimer_v2",
                            "alphafold2_multimer_v3",
                        ],
                        )
    parser.add_argument("--amber",
                        default=False,
                        action="store_true",
                        help="Use amber for structure refinement."
                             "To control number of top ranked structures are relaxed set --num-relax.",
                        )
    parser.add_argument("--num-relax",
                        help="specify how many of the top ranked structures to relax using amber.",
                        type=int,
                        default=0,
                        )
    parser.add_argument("--templates", default=False, action="store_true", help="Use templates from pdb")
    parser.add_argument("--custom-template-path",
                        type=str,
                        default=None,
                        help="Directory with pdb files to be used as input",
                        )
    parser.add_argument("--rank",
                        help="rank models by auto, plddt or ptmscore",
                        type=str,
                        default="auto",
                        choices=["auto", "plddt", "ptm", "iptm", "multimer"],
                        )
    parser.add_argument("--pair-mode",
                        help="rank models by auto, unpaired, paired, unpaired_paired",
                        type=str,
                        default="unpaired_paired",
                        choices=["unpaired", "paired", "unpaired_paired"],
                        )
    parser.add_argument("--sort-queries-by",
                        help="sort queries by: none, length, random",
                        type=str,
                        default="length",
                        choices=["none", "length", "random"],
                        )
    parser.add_argument("--save-single-representations",
                        default=False,
                        action="store_true",
                        help="saves the single representation embeddings of all models",
                        )
    parser.add_argument("--save-pair-representations",
                        default=False,
                        action="store_true",
                        help="saves the pair representation embeddings of all models",
                        )
    parser.add_argument("--use-dropout",
                        default=False,
                        action="store_true",
                        help="activate dropouts during inference to sample from uncertainity of the models",
                        )
    parser.add_argument("--max-seq",
                        help="number of sequence clusters to use",
                        type=int,
                        default=None,
                        )
    parser.add_argument("--max-extra-seq",
                        help="number of extra sequences to use",
                        type=int,
                        default=None,
                        )
    parser.add_argument("--max-msa",
                        help="defines: `max-seq:max-extra-seq` number of sequences to use",
                        type=str,
                        default=None,
                        )
    parser.add_argument("--disable-cluster-profile",
                        default=False,
                        action="store_true",
                        help="EXPERIMENTAL: for multimer models, disable cluster profiles",
                        )
    parser.add_argument("--zip",
                        default=False,
                        action="store_true",
                        help="zip all results into one <jobname>.result.zip and delete the original files",
                        )
    parser.add_argument("--use-gpu-relax",
                        default=False,
                        action="store_true",
                        help="run amber on GPU instead of CPU",
                        )
    parser.add_argument("--save-all",
                        default=False,
                        action="store_true",
                        help="save ALL raw outputs from model to a pickle file",
                        )
    parser.add_argument("--save-recycles",
                        default=False,
                        action="store_true",
                        help="save all intermediate predictions at each recycle",
                        )
    parser.add_argument("--overwrite-existing-results", default=False, action="store_true")
    parser.add_argument("--disable-unified-memory",
                        default=False,
                        action="store_true",
                        help="if you are getting tensorflow/jax errors it might help to disable this",
                        )
    parser.add_argument("--disulfide-bond-pairs", type=int, nargs='+', default=[])
    parser.add_argument("--flag-cyclic-peptide", type=int, nargs='+', default=[],
                        help="length is the number of chains. 1 for cyclic peptide, 0 for linear peptide. "
                             "For example, [0 1 1] denotes the 2nd chain and the 3rd chain are cyclic peptides. ")
    parser.add_argument("--flag-nc", type=int, nargs='+', default=[],
                        help="length is the number of chains. 1 for cyclic peptide by nc, 0 for not. "
                             "For example, [0 0 1] denotes the 3rd chain is cyclic peptide by nc. ")
    parser.add_argument('--index-ss', type=int, nargs='+', action='append', default=[],
                        help='the indices start at 0, the first number is chain_id, '
                             'and the following numbers are the indices of disulfide bond pairs. For example, '
                             '[[2 0 7][1 2 8]] denotes the 3rd chain and the 2nd chain are cyclic peptide by ss. '
                             'In the 3rd chain, the 1st amino acid and the 8th amino acid are linked by ss. '
                             'In the 2nd chain, the 3rd amino acid and the 9th amino acid are linked by ss. ')
    parser.add_argument("--custom-dist-cst", type=int, nargs='+', action='append', default=[],
                        help="the indices start at 0, "
                             "the first number is chain_id, "
                             "the second number is the type of constraints, 1 for head-to-tail, 2 for disulfide bridges, "
                             "the third and fourth number is the indices of amino acid pairs, "
                             "the five number is the distance in nanometer*100."
                             "For example, [[2 2 0 7 15]] denotes the distance between the two sulphur atoms "
                             "in amino acid 0 and 7 of peptide chain 2 is constrained to 0.15 nanometer. "
                             "For another example, [[3 1 0 7 16]] denotes the distance between n-terminal of amino acid 0"
                             "and c-terminal pf amino acid 7 in peptide chain 3 is constrained to 0.16 nanometer. "
                             "Note that the length of peptide chain 3 is 8. ")
    parser.add_argument("--unnatural_residue", type=str, nargs='+', default=[],
                        help="the order of unnatural residue. For example, ABA PTR")

    args = parser.parse_args()

    disulfide_bond_pairs = []
    for i in range(0, len(args.disulfide_bond_pairs), 2):
        disulfide_bond_pairs.append((args.disulfide_bond_pairs[i], args.disulfide_bond_pairs[i + 1]))

    index_ss = {}
    for i in range(len(args.index_ss)):
        pair_ss = []
        for j in range(1, len(args.index_ss[i]), 2):
            pair_ss.append((args.index_ss[i][j], args.index_ss[i][j+1]))
        index_ss['c'+str(args.index_ss[i][0])] = pair_ss

    # disable unified memory
    if args.disable_unified_memory:
        for k in ENV.keys():
            if k in os.environ: del os.environ[k]

    setup_logging(Path(args.results).joinpath("log.txt"))

    version = importlib_metadata.version("colabfold")
    commit = get_commit()
    if commit:
        version += f" ({commit})"

    logger.info(f"Running colabfold {version}")

    data_dir = Path(args.data or default_data_dir)

    queries, is_complex = get_queries(args.input, args.sort_queries_by)
    model_type = set_model_type(is_complex, args.model_type)

    # download_alphafold_params(model_type, data_dir)

    if args.msa_mode != "single_sequence" and not args.templates:
        uses_api = any((query[2] is None for query in queries))
        if uses_api and args.host_url == DEFAULT_API_SERVER:
            print(ACCEPT_DEFAULT_TERMS, file=sys.stderr)

    model_order = [int(i) for i in args.model_order.split(",")]

    assert args.recompile_padding >= 0, "Can't apply negative padding"

    # backward compatibility
    if args.amber and args.num_relax == 0:
        args.num_relax = args.num_models * args.num_seeds

    run(
        unnatural_residue=args.unnatural_residue,
        disulfide_bond_pairs=disulfide_bond_pairs,
        flag_cyclic_peptide=args.flag_cyclic_peptide,
        flag_nc=args.flag_nc,
        index_ss=index_ss,
        dist_cst_ss=args.custom_dist_cst,
        queries=queries,
        result_dir=args.results,
        use_templates=args.templates,
        custom_template_path=args.custom_template_path,
        num_relax=args.num_relax,
        msa_mode=args.msa_mode,
        model_type=model_type,
        num_models=args.num_models,
        num_recycles=args.num_recycle,
        recycle_early_stop_tolerance=args.recycle_early_stop_tolerance,
        num_ensemble=args.num_ensemble,
        model_order=model_order,
        is_complex=is_complex,
        keep_existing_results=not args.overwrite_existing_results,
        rank_by=args.rank,
        pair_mode=args.pair_mode,
        data_dir=data_dir,
        host_url=args.host_url,
        random_seed=args.random_seed,
        num_seeds=args.num_seeds,
        stop_at_score=args.stop_at_score,
        recompile_padding=args.recompile_padding,
        zip_results=args.zip,
        save_single_representations=args.save_single_representations,
        save_pair_representations=args.save_pair_representations,
        use_dropout=args.use_dropout,
        max_seq=args.max_seq,
        max_extra_seq=args.max_extra_seq,
        max_msa=args.max_msa,
        use_cluster_profile=not args.disable_cluster_profile,
        use_gpu_relax=args.use_gpu_relax,
        save_all=args.save_all,
        save_recycles=args.save_recycles,
    )


if __name__ == "__main__":
    main()
