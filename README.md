# Predicting the structures of cyclic peptides containing unnatural amino acids by HighFold2

![overview](./util/overview.png)HighFold2 can accurately predict the structures of cyclic peptides containing unnatural amino acids and their complexes. It involves training a deep learning model based on AlphaFold-Multimer using linear peptide structures with unnatural amino acids, then modifying the model's relative position encoding matrix, enabling it to predict the cyclic peptide structures successfully. Then, relaxation is performed to refine the spatial structure further. We believe that this method will serve as a powerful tool for the development of cyclic peptide-based therapeutics.

## Installation

You should first download the source code and dependencies as follows:

```bash
git clone https://github.com/hongliangduan/HighFold2.git
cd HighFold2
conda create --name highfold2 --file requirements_conda.txt --channel bioconda
conda activate highfold2
pip install -r requirements_pip.txt
pip install -e ./colabfold
pip install -e ./alphafold
```

## Usage

### Prediction

For the prediction, you should download our fine-tuned parameters from [onedrive](https://1drv.ms/f/c/a6a575f7399b61f9/Ejavf9uTnRBHhQAVQWKhJ7wByd57xfQALTpMgoZqVXnmBg?e=Ij1NaR) and put them in the fine_tuning/params folder. Then, you should prepare the fasta file for your sequence, all the unnatural residues are replaced with 'X' in fasta files. Finally, enter a command similar to the following in the terminal, we also provide some detailed prediction examples in the prediction.ipynb.

```bash
python prediction.py --model-type alphafold2_multimer_v3 {fasta_path} {output_path} --unnatural_residue {unnatural_amino_acids} --flag-cyclic-peptide 1 --flag-nc 1 --amber
```

All the flags useful for prediction can be seen by typing the following command in the terminal:

```bash
python prediction.py -h
```

### Training

For the training, you should obtain the feature and AlphaFold-Multimer's parameters firstly as follows:

```bash
python gen_feature.py
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
mkdir -p alphafold_multimer/params
tar -xvf alphafold_params_2022-12-06.tar -C alphafold_multimer/params
```

After getting the feature and parameters, you can run the training.py as follows:

```bash
python training.py
```
