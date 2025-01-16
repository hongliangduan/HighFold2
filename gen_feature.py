from colabfold.batch import (
                            get_queries,
                            get_msa_and_templates,
                            unserialize_msa,
                            msa_to_str,
                            generate_input_feature)
from pathlib import Path
from alphafold.common import residue_constants
from colabfold.utils import DEFAULT_API_SERVER,safe_filename
import numpy as np
import pandas as pd
import os

def generate_feature(fasta_path, result_dir, unnatural):
    args_disulfide_bond_pairs = []
    disulfide_bond_pairs = []
    for i in range(0, len(args_disulfide_bond_pairs), 2):
        disulfide_bond_pairs.append((args_disulfide_bond_pairs[i], args_disulfide_bond_pairs[i + 1]))
    flag_cyclic_peptide  = []
    flag_nc = []
    index_ss = []

    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    queries, is_complex = get_queries(fasta_path, sort_queries_by='length')
    model_type = 'alphafold2_multimer_v3'

    msa_mode = 'mmseqs2_uniref_env'
    templates = False
    host_url = DEFAULT_API_SERVER
    if msa_mode != "single_sequence" and not templates:
        uses_api = any((query[2] is None for query in queries))

    max_len = 0
    max_num = 0
    for _, query_sequence, _ in queries:
        N = 1 if isinstance(query_sequence, str) else len(query_sequence)
        L = len("".join(query_sequence))
        if L > max_len: max_len = L
        if N > max_num: max_num = N
    set_if = lambda x, y: y if x is None else x
    max_seq = None
    max_extra_seq = None
    (max_seq, max_extra_seq) = (set_if(max_seq, 508), set_if(max_extra_seq, 2048))

    keep_existing_results = True
    for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(queries):
        jobname = safe_filename(raw_jobname)
        result_zip = result_dir.joinpath(jobname).with_suffix(".result.zip")
        if keep_existing_results and result_zip.is_file():
            continue
        is_done_marker = result_dir.joinpath(jobname + ".done.txt")
        if keep_existing_results and is_done_marker.is_file():
            continue

        seq_len = len("".join(query_sequence))

    use_templates = False
    custom_template_path = None
    pair_mode = 'unpaired_paired'
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



    (feature_dict, domain_names) \
        = generate_input_feature(disulfide_bond_pairs, flag_cyclic_peptide, flag_nc, index_ss,
                                         query_seqs_unique, query_seqs_cardinality, unpaired_msa, paired_msa,
                                         template_features, is_complex, model_type, max_seq=max_seq)

    indices = np.where(feature_dict['aatype'] == len(residue_constants.restypes))[0]
    if model_type == 'alphafold2_multimer_v3':
        assert len(indices) == len(unnatural)
        for i, name in enumerate(unnatural):
            feature_dict['aatype'][indices[i]] = residue_constants.restypes.index(name)

    np.save(result_dir.joinpath(jobname + ".npy"), feature_dict)

def get_all_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

def find_non_natural_aa(pdbid):
    result = df[df['pdbid'] == pdbid]
    if not result.empty:
        return result['non_natural_aa'].values[0]
    else:
        return None

fasta_path = './fasta'
df = pd.read_csv('./seq.csv')
fasta = get_all_files(fasta_path)
for i in fasta:
    pdbid = i[-10:-6]
    print(pdbid)
    non_natural_aa = find_non_natural_aa(pdbid)
    non_natural_aa = non_natural_aa.split(':')
    result_dir = './feature/'+ pdbid
    generate_feature(i, result_dir, non_natural_aa)