# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cleans up a PDB file using pdbfixer in preparation for OpenMM simulations.

fix_pdb uses a third-party tool. We also support fixing some additional edge
cases like removing chains of length one (see clean_structure).
"""
import io

import pdbfixer
from simtk.openmm import app
from simtk.openmm.app import element
import subprocess
import os


def fix_pdb(pdbfile, alterations_info, dist_cst_ss, disulf, aa_names, nc, cp):
  """Apply pdbfixer to the contents of a PDB file; return a PDB string result.

  1) Replaces nonstandard residues.
  2) Removes heterogens (non protein residues) including water.
  3) Adds missing residues and missing atoms within existing residues.
  4) Adds hydrogens assuming pH=7.0.
  5) KeepIds is currently true, so the fixer must keep the existing chain and
     residue identifiers. This will fail for some files in wider PDB that have
     invalid IDs.

  Args:
    pdbfile: Input PDB file handle.
    alterations_info: A dict that will store details of changes made.

  Returns:
    A PDB string representing the fixed structure.
  """
  for i in disulf:
    for j in i:
      aa_names[j] = 'CYX'
  temp_str = 'temp = { '
  for i in aa_names:
    temp_str = temp_str + str(i)+ ' '
  temp_str = temp_str + '}\n'
  with open('tmp.pdb', 'w') as tmp_file:
    tmp_file.write(pdbfile.getvalue())
  
  
  with open('leap.in', 'w') as f:
    f.write('source leaprc.protein.ff14SB\n')

    f.write('loadAmberPrep ./forcefield/2MR.prepin\n')
    f.write('loadAmberParams ./forcefield/2MR.frcmod\n')

    f.write('loadAmberPrep ./forcefield/ABA.prepin\n')
    f.write('loadAmberParams ./forcefield/ABA.frcmod\n')

    f.write('loadAmberPrep ./forcefield/AIB.prepin\n')
    f.write('loadAmberParams ./forcefield/AIB.frcmod\n')

    f.write('loadAmberPrep ./forcefield/ALC.prepin\n')
    f.write('loadAmberParams ./forcefield/ALC.frcmod\n')

    f.write('loadAmberPrep ./forcefield/ALY.prepin\n')
    f.write('loadAmberParams ./forcefield/ALY.frcmod\n')

    f.write('loadAmberPrep ./forcefield/CSO.prepin\n')
    f.write('loadAmberParams ./forcefield/CSO.frcmod\n')

    f.write('loadAmberPrep ./forcefield/DA2.prepin\n')
    f.write('loadAmberParams ./forcefield/DA2.frcmod\n')

    f.write('loadAmberPrep ./forcefield/DLE.prepin\n')
    f.write('loadAmberParams ./forcefield/DLE.frcmod\n')

    f.write('loadAmberPrep ./forcefield/DPR.prepin\n')
    f.write('loadAmberParams ./forcefield/DPR.frcmod\n')

    f.write('loadAmberPrep ./forcefield/HYP.prepin\n')
    f.write('loadAmberParams ./forcefield/HYP.frcmod\n')

    f.write('loadAmberPrep ./forcefield/M3L.prepin\n')
    f.write('loadAmberParams ./forcefield/M3L.frcmod\n')

    f.write('loadAmberPrep ./forcefield/MLY.prepin\n')
    f.write('loadAmberParams ./forcefield/MLY.frcmod\n')

    f.write('loadAmberPrep ./forcefield/MLZ.prepin\n')
    f.write('loadAmberParams ./forcefield/MLZ.frcmod\n')

    f.write('loadAmberPrep ./forcefield/MSE.prepin\n')
    f.write('loadAmberParams ./forcefield/MSE.frcmod\n')

    f.write('loadAmberPrep ./forcefield/NLE.prepin\n')
    f.write('loadAmberParams ./forcefield/NLE.frcmod\n')

    f.write('loadAmberPrep ./forcefield/NVA.prepin\n')
    f.write('loadAmberParams ./forcefield/NVA.frcmod\n')

    f.write('loadAmberPrep ./forcefield/ORN.prepin\n')
    f.write('loadAmberParams ./forcefield/ORN.frcmod\n')

    f.write('loadAmberPrep ./forcefield/PCA.prepin\n')
    f.write('loadAmberParams ./forcefield/PCA.frcmod\n')

    f.write('loadAmberPrep ./forcefield/PRK.prepin\n')
    f.write('loadAmberParams ./forcefield/PRK.frcmod\n')

    f.write('loadAmberPrep ./forcefield/PTR.prepin\n')
    f.write('loadAmberParams ./forcefield/PTR.frcmod\n')

    f.write('loadAmberPrep ./forcefield/SEP.prepin\n')
    f.write('loadAmberParams ./forcefield/SEP.frcmod\n')

    f.write('loadAmberPrep ./forcefield/TPO.prepin\n')
    f.write('loadAmberParams ./forcefield/TPO.frcmod\n')
    if cp:
      if aa_names[-1] == 'NH2':
        f.write(temp_str.replace('NH2', 'NHE'))
        f.write('mol = loadPDBusingseq tmp.pdb temp\n')
      else:
        f.write(temp_str)
        f.write('mol = loadPDBusingseq tmp.pdb temp\n')
      for i in disulf:
        f.write(f'bond mol.{i[0]+1}.SG mol.{i[1]+1}.SG\n')
      for i in nc:
        f.write(f'bond mol.{i[0]+1}.N mol.{i[1]+1}.C\n')
    elif aa_names[-1] == 'NH2':
      f.write(temp_str.replace('NH2', 'NHE'))
      f.write('mol = loadPDBusingseq tmp.pdb temp\n')
    else:
      f.write('mol = loadPdb tmp.pdb\n')
    f.write('check mol\n')
    f.write('saveAmberParm mol system.prmtop system.inpcrd\n')
    f.write('savepdb mol tmp.pdb\n')
    f.write('quit')
  # os.system('tleap -f leap.in')
  # fixer = pdbfixer.PDBFixer(filename='tmp.pdb')
  with open(os.devnull, 'w') as FNULL:
    subprocess.call(['tleap', '-f', 'leap.in'], stdout=FNULL, stderr=FNULL)
  fixer = pdbfixer.PDBFixer(filename='tmp.pdb')
  # fixer = pdbfixer.PDBFixer(pdbfile=pdbfile)
  # fixer.findNonstandardResidues()
  # alterations_info['nonstandard_residues'] = fixer.nonstandardResidues
  # fixer.replaceNonstandardResidues()
  # _remove_heterogens(fixer, alterations_info, keep_water=False)
  fixer.findMissingResidues()
  alterations_info['missing_residues'] = fixer.missingResidues
  fixer.findMissingAtoms()
  alterations_info['missing_heavy_atoms'] = fixer.missingAtoms
  alterations_info['missing_terminals'] = fixer.missingTerminals
  fixer.addMissingAtoms(seed=0)
  # fixer.addMissingHydrogens()
  out_handle = io.StringIO()
  app.PDBFile.writeFile(fixer.topology, fixer.positions, out_handle,
                        keepIds=True)
  return out_handle.getvalue()


def clean_structure(pdb_structure, alterations_info):
  """Applies additional fixes to an OpenMM structure, to handle edge cases.

  Args:
    pdb_structure: An OpenMM structure to modify and fix.
    alterations_info: A dict that will store details of changes made.
  """
  _replace_met_se(pdb_structure, alterations_info)
  _remove_chains_of_length_one(pdb_structure, alterations_info)


def _remove_heterogens(fixer, alterations_info, keep_water):
  """Removes the residues that Pdbfixer considers to be heterogens.

  Args:
    fixer: A Pdbfixer instance.
    alterations_info: A dict that will store details of changes made.
    keep_water: If True, water (HOH) is not considered to be a heterogen.
  """
  initial_resnames = set()
  for chain in fixer.topology.chains():
    for residue in chain.residues():
      initial_resnames.add(residue.name)
  fixer.removeHeterogens(keepWater=keep_water)
  final_resnames = set()
  for chain in fixer.topology.chains():
    for residue in chain.residues():
      final_resnames.add(residue.name)
  alterations_info['removed_heterogens'] = (
      initial_resnames.difference(final_resnames))


def _replace_met_se(pdb_structure, alterations_info):
  """Replace the Se in any MET residues that were not marked as modified."""
  modified_met_residues = []
  for res in pdb_structure.iter_residues():
    name = res.get_name_with_spaces().strip()
    if name == 'MET':
      s_atom = res.get_atom('SD')
      if s_atom.element_symbol == 'Se':
        s_atom.element_symbol = 'S'
        s_atom.element = element.get_by_symbol('S')
        modified_met_residues.append(s_atom.residue_number)
  alterations_info['Se_in_MET'] = modified_met_residues


def _remove_chains_of_length_one(pdb_structure, alterations_info):
  """Removes chains that correspond to a single amino acid.

  A single amino acid in a chain is both N and C terminus. There is no force
  template for this case.

  Args:
    pdb_structure: An OpenMM pdb_structure to modify and fix.
    alterations_info: A dict that will store details of changes made.
  """
  removed_chains = {}
  for model in pdb_structure.iter_models():
    valid_chains = [c for c in model.iter_chains() if len(c) > 1]
    invalid_chain_ids = [c.chain_id for c in model.iter_chains() if len(c) <= 1]
    model.chains = valid_chains
    for chain_id in invalid_chain_ids:
      model.chains_by_id.pop(chain_id)
    removed_chains[model.number] = invalid_chain_ids
  alterations_info['removed_chains'] = removed_chains
