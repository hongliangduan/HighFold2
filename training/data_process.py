import torch
import numpy as np
from alphafold.data import pipeline, parsers
from alphafold.common import residue_constants
from utils import load_pdb_coords, pseudo_beta_fn_np, fill_af_coords, cropcontiguous, chain_lens, cropspatial, get_ca_coordinates, smiles_dict, get_atom_features, get_bond_feat, get_atom_feat, get_mask


class CustomPandasDataset(torch.utils.data.Dataset):
    def __init__(self, df, fn, crop_size=0):
        self.df = df
        self.fn = fn
        self.crop_size = crop_size
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return self.fn(row, self.crop_size)
    

def create_batch_from_dataset(row,crop_size):
    batch = np.load(row['feature_dict'], allow_pickle=True).item()
    print(row['pdbid'])
    chains, all_resids, all_coords, all_name1s = load_pdb_coords(row['native_pdb'])
    all_positions, all_positions_mask = fill_af_coords(chains, all_resids, all_coords)
    batch['all_atom_positions'] = all_positions
    batch['all_atom_mask'] = all_positions_mask

    if len(batch['aatype']) > crop_size and crop_size:
    # if crop_size:
        print('crop')
        nk = chain_lens(batch['residue_index'])
        # flag = np.random.randint(0, 2)
        # if flag == 0:
        if True:
            seq_mask = cropcontiguous(nk, crop_size)
        # else:
        #     if len(nk) >= 2:
        #         del batch['seq_mask']
        #         ca_coordinates = get_ca_coordinates(row['native_pdb'])
        #         interface_idx = np.random.randint(nk[0], nk[0]+nk[1])
        #         seq_mask = cropspatial(ca_coordinates, interface_idx, crop_size)
        #         batch['seq_mask'] = np.array(seq_mask)

            batch['all_atom_positions'] = batch['all_atom_positions'][seq_mask == 1]
            batch['all_atom_mask'] = batch['all_atom_mask'][seq_mask == 1]
            batch['aatype'] = batch['aatype'][seq_mask == 1]
            batch['msa'] = batch['msa'][:,seq_mask == 1]
            batch['msa_mask'] = batch['msa_mask'][:,seq_mask == 1]
            batch['bert_mask'] = batch['bert_mask'][:,seq_mask == 1]
            batch['deletion_matrix'] = batch['deletion_matrix'][:,seq_mask == 1]
            batch['asym_id'] = batch['asym_id'][seq_mask == 1]
            batch['entity_id'] = batch['entity_id'][seq_mask == 1]
            batch['sym_id'] = batch['sym_id'][seq_mask == 1]
            batch['template_aatype'] = batch['template_aatype'][:,seq_mask == 1]
            batch['template_all_atom_positions'] = batch['template_all_atom_positions'][:,seq_mask == 1,:,:]
            batch['template_all_atom_mask'] = batch['template_all_atom_mask'][:,seq_mask == 1,:]
            batch['seq_mask'] = batch['seq_mask'][seq_mask == 1]

            arr = batch['residue_index'][seq_mask == 1]
            result = np.zeros_like(arr)
            start = 0
            while start < len(arr):
                end = start
                while end + 1 < len(arr) and arr[end + 1] == arr[end] + 1:
                    end += 1
                min_val = arr[start]
                result[start:end + 1] = arr[start:end + 1] - min_val
                start = end + 1
            batch['residue_index'] = result

            batch['seq_length'] = len(batch['aatype'])
            batch['deletion_mean'] = batch['deletion_mean'][seq_mask == 1]
            batch['entity_mask'] = batch['entity_mask'][seq_mask == 1]

            array = batch['residue_index']
            breakpoints = [0] + [i for i in range(1, len(array)) if array[i] < array[i-1]] + [len(array)]
            group_lengths = [breakpoints[i+1] - breakpoints[i] for i in range(len(breakpoints)-1)]
            batch['chain_length'] = group_lengths

    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn_np(batch['aatype'], batch['all_atom_positions'], batch['all_atom_mask'])
    batch['pseudo_beta'] = np.array(pseudo_beta)[None,]
    batch['pseudo_beta_mask'] = np.array(pseudo_beta_mask)[None,]
    batch['resolution'] = np.array(1.0)[None,]
    batch['pdbid'] = row['pdbid']
    batch['is_continuous'] = row['is_continuous']

    indices = [i for i, arr in enumerate(batch['chain_length']) if arr < 50]
    ligand_feats_dict = {}
    for index in indices:
        end = 0
        for i in range(index+1):
            end += batch['chain_length'][i]
        start = end - batch['chain_length'][index]
        pep_index = batch['aatype']
        pep_index = batch['aatype'][start:end]
        pep_chara = [residue_constants.resnames[i] for i in pep_index]
        # print(pep_chara)
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
    batch['ligand_feats_dict'] = ligand_feats_dict

    return batch