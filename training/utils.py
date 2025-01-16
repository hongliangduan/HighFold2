import numpy as np
import os
from alphafold.common import residue_constants
from alphafold.common import protein
import jax
import jax.numpy as jnp
from Bio import PDB
from rdkit import Chem
from rdkit.Chem import AllChem
import copy

def load_pdb_coords(pdbfile):
    chains = []
    all_resids = {}
    all_coords = {}
    all_name1s = {}

    with open(pdbfile,'r') as data:
        for line in data:
            if (line[:6] in ['ATOM  ','HETATM'] and line[17:20] != 'HOH'):
                if line[17:20] in residue_constants.restype_3to1:
                    name1 = residue_constants.restype_3to1[line[17:20]]
                    resid = line[22:27]
                    chain = line[21]
                    if chain not in all_resids:
                        all_resids[chain] = []
                        all_coords[chain] = {}
                        all_name1s[chain] = {}
                        chains.append(chain)
    
                    atom = line[12:16].split()[0]
                    if resid not in all_resids[chain]:
                        all_resids[chain].append(resid)
                        all_coords[chain][resid] = {}
                        all_name1s[chain][resid] = name1

                    all_coords[chain][resid][atom] = np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])])

    # check for chainbreaks
    maxdis = 1.75
    for chain in chains:
        for res1, res2 in zip(all_resids[chain][:-1], all_resids[chain][1:]):
            coords1 = all_coords[chain][res1]
            coords2 = all_coords[chain][res2]
            if 'C' in coords1 and 'N' in coords2:
                dis = np.sqrt(np.sum(np.square(coords1['C']-coords2['N'])))
                if dis>maxdis:
                    print('WARNING chainbreak:', chain, res1, res2, dis, pdbfile)

    return chains, all_resids, all_coords, all_name1s



def pseudo_beta_fn_np(aatype, all_atom_positions, all_atom_masks):
  """Create pseudo beta features."""

  is_gly = np.equal(aatype, residue_constants.restype_order['G'])
  ca_idx = residue_constants.atom_order['CA']
  cb_idx = residue_constants.atom_order['CB']
  pseudo_beta = np.where(
      np.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
      all_atom_positions[..., ca_idx, :],
      all_atom_positions[..., cb_idx, :])

  if all_atom_masks is not None:
    pseudo_beta_mask = np.where(
        is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
    pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
    return pseudo_beta, pseudo_beta_mask
  else:
    return pseudo_beta


def fill_af_coords(chain_order,all_resids,all_coords):

    crs = [(chain,resid) for chain in chain_order for resid in all_resids[chain]]
    num_res = len(crs)
    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],dtype=np.int64)
    for res_index, (chain,resid) in enumerate(crs):
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        for atom_name, xyz in all_coords[chain][resid].items():
            x,y,z = xyz
            if atom_name in residue_constants.atom_order.keys():
                pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                mask[residue_constants.atom_order[atom_name]] = 1.0
            else:
                name = atom_name[:]
                while name[0] in '123':
                    name = name[1:]
                if name[0] != 'H':
                    print('unrecognized atom:', atom_name, chain, resid)

        all_positions[res_index] = pos
        all_positions_mask[res_index] = mask
    return all_positions, all_positions_mask



def flatten_dict(d, parent_key='', sep='//'):

    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def norm_grads_per_example(grads, l2_norm_clip):
    nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads)
    a = [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads]
    total_grad_norm = jnp.linalg.norm(a[0])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    grads = jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)
    return grads


def collate_fn(batch):
    for sample in batch:
        return sample
    
def save_pdb(predicted_dict, batch, save_path, epoch, pdbid):
    final_atom_mask = predicted_dict["structure_module"]["final_atom_mask"]
    b_factors = predicted_dict["plddt"][:, None] * final_atom_mask
    unrelaxed_protein = protein.from_prediction(features=batch,result=predicted_dict,b_factors=b_factors,remove_leading_feature_dimension=("multimer" not in 'alphafold2_multimer_v3'))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    outfile = save_path+f'/valid_result_{epoch}_{pdbid}.pdb'
    with open(outfile, 'w') as f:
        f.write(protein.to_pdb(unrelaxed_protein))


def cropcontiguous(nk, N_res):
    n_added = 0
    n_remaining = N_res
    N_chains = len(nk)
    chains = sorted(enumerate(nk), key=lambda x: x[1])
    
    masks = [None] * N_chains
    
    for idx, length in chains:
        n_remaining -= length
        
        crop_size_max = min(N_res - n_added, length)
        crop_size_min = min(length, max(0, N_res - (n_added + n_remaining)))
        
        if crop_size_min > crop_size_max:
            crop_size_min, crop_size_max = crop_size_max, crop_size_min
        
        crop_size = crop_size_min
        n_added += crop_size
        
        crop_start = np.random.randint(0, length - crop_size + 1)
        mk = np.zeros(length, dtype=float)
        mk[crop_start:crop_start + crop_size] = 1
        masks[idx] = mk

    masks = np.concatenate(masks)
    return masks


def chain_lens(arr):
    chain_len = []
    for i in range(len(arr)-1):
        if arr[i] - arr[i+1] != -1:
            chain_len.append(arr[i]+1)
    chain_len.append(arr[-1]+1)
    return chain_len

import numpy as np

def cropspatial(xi, c, N_res):
    if N_res >= len(xi):
        return np.ones(len(xi))
    else:
        di = np.linalg.norm(xi - xi[c], axis=1) + np.arange(len(xi)) * 1e-3
        d_cutoff = np.partition(di, N_res - 1)[N_res - 1]
        mi = di <= d_cutoff
        mi = mi.astype(float)
        return mi

def get_ca_coordinates(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('X', pdb_file)
    
    ca_coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].coord)
                    
    return np.array(ca_coords)

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

    has_bond = copy.deepcopy(bond_matrix)
    has_bond[has_bond>0]=1
    return np.array(atom_types), np.array(atoms), bond_matrix, has_bond

def get_atom_features(smiles_string):

    input_smiles = smiles_string
    #Atom encoding - no hydrogens
    atom_encoding = {'B':0, 'C':1, 'F':2, 'I':3, 'N':4, 'O':5, 'P':6, 'S':7,'Br':8, 'Cl':9, #Individual encoding
                    'As':10, 'Co':10, 'Fe':10, 'Mg':10, 'Pt':10, 'Rh':10, 'Ru':10, 'Se':10, 'Si':10, 'Te':10, 'V':10, 'Zn':10 #Joint (rare)
                    }

    atom_types, atoms, bond_types, bond_mask = bonds_from_smiles(input_smiles, atom_encoding)

    ligand_feats = {}
    ligand_feats['atoms'] = atoms
    ligand_feats['atom_types'] = atom_types
    ligand_feats['bond_types'] = bond_types
    ligand_feats['bond_mask'] = bond_mask
    return ligand_feats

def get_bond_feat(ligand_feats):
    atom_len = len(ligand_feats['atoms'])
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


def del_pdbs(directory, global_steps):
    global_steps = str(global_steps)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filepath.split('_')[-2] != global_steps:
            os.remove(filepath)