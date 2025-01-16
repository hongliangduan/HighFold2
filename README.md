# Predicting the structures of cyclic peptides containing unnatural amino acids by HighFold2

![overview](./util/overview.png)HighFold2 can accurately predict the structures of cyclic peptides containing unnatural amino acids and their complexes. It involves training a deep learning model based on AlphaFold-Multimer using linear peptide structures with unnatural amino acids, then modifying the model's relative position encoding matrix, enabling it to predict the cyclic peptide structures successfully. Then, relaxation is performed to refine the spatial structure further. We believe that this method will serve as a powerful tool for the development of cyclic peptide-based therapeutics.

## Installation

HigFold2 is based on LocalColabFold, which is available at https://github.com/YoshitakaMo/localcolabfold, so you should install it first. Then, you should download our source code as follows:

```bash
git clone https://github.com/hongliangduan/HighFold2.git
cd HighFold2
```

After, you should download the parameters from [onedrive](https://1drv.ms/f/c/a6a575f7399b61f9/Ejavf9uTnRBHhQAVQWKhJ7wBzV7-vsIecRnZ3DovnfV_Cg?e=nbO0N3), and put them in the colabfold/params folder. Finally, you should replace the original alphafold and colabfold folders in /localcolabfold/colabfold-conda/lib/python3.10/site-packages/ with the ones from the modify\_code folder. For relaxation and training, you should also install some other packages. All packages we used can be seen in the requirement.txt.

## Usage

### Prediction

For the prediction, you should be in the modified locallocalfold environment and the HighFold folder. Then, you should prepare the fasta file for your sequence, all the unnatural residues are replaced with 'X' in fasta files. Finally, enter a command similar to the following in the terminal:

```bash
colabfold_batch --model-type alphafold2_multimer_v3 {fasta_path} {output_path} --unnatural_residue {unnatural_amino_acids} --flag-cyclic-peptide 1 --flag-nc 1 --amber
```

All the flags useful for prediction can be seen by typing the following command in the terminal:

```bash
colabfold_batch -h
```

### Training

For the training, you should obtain the feature firstly as follows:

```bash
python gen_feature.py
```

After getting the feature, you should change the directory to the training folder and copy the original AlphaFold-Multimer parameters to the colabfold folder, then run the training.py as follows:

```bash
cd training
python training.py
```
