from typing import Dict, Tuple, Union
import os
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import GetMolFrags, FragmentOnBonds
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

"""
Taken from https://github.com/dptech-corp/Uni-Dock/blob/027a13f05007ce474c7a83d547b968938f81cea0/unidock_tools/src/unidock_tools/modules/ligand_prep/torsion_tree.py
"""

ATOM_TYPE_DEFINITION_LIST = [{'smarts': '[#1]', 'atype': 'H', 'comment': 'invisible'},
                             {'smarts': '[#1][#7,#8,#9,#15,#16]', 'atype': 'HD', 'comment': None},
                             {'smarts': '[#5]', 'atype': 'B', 'comment': None},
                             {'smarts': '[C]', 'atype': 'C', 'comment': None},
                             {'smarts': '[c]', 'atype': 'A', 'comment': None},
                             {'smarts': '[#7]', 'atype': 'NA', 'comment': None},
                             {'smarts': '[#8]', 'atype': 'OA', 'comment': None},
                             {'smarts': '[#9]', 'atype': 'F', 'comment': None},
                             {'smarts': '[#12]', 'atype': 'Mg', 'comment': None},
                             {'smarts': '[#14]', 'atype': 'Si', 'comment': None},
                             {'smarts': '[#15]', 'atype': 'P', 'comment': None},
                             {'smarts': '[#16]', 'atype': 'S', 'comment': None},
                             {'smarts': '[#17]', 'atype': 'Cl', 'comment': None},
                             {'smarts': '[#20]', 'atype': 'Ca', 'comment': None},
                             {'smarts': '[#25]', 'atype': 'Mn', 'comment': None},
                             {'smarts': '[#26]', 'atype': 'Fe', 'comment': None},
                             {'smarts': '[#30]', 'atype': 'Zn', 'comment': None},
                             {'smarts': '[#35]', 'atype': 'Br', 'comment': None},
                             {'smarts': '[#53]', 'atype': 'I', 'comment': None},
                             {'smarts': '[#7X3v3][a]', 'atype': 'N', 'comment': 'pyrrole, aniline'},
                             {'smarts': '[#7X3v3][#6X3v4]', "atype": 'N', 'comment': 'amide'},
                             {'smarts': '[#7+1]', 'atype': 'N', 'comment': 'ammonium, pyridinium'},
                             {'smarts': '[SX2]', 'atype': 'SA', 'comment': 'sulfur acceptor'}]


def assign_atom_properties(mol: Chem.Mol):
    atom_positions = mol.GetConformer().GetPositions()
    num_atoms = mol.GetNumAtoms()

    internal_atom_idx = 0
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetIntProp('sdf_atom_idx', atom_idx + 1)
        if not atom.HasProp('atom_name'):
            atom_element = atom.GetSymbol()
            atom_name = atom_element + str(internal_atom_idx + 1)
            atom.SetProp('atom_name', atom_name)
            atom.SetProp('residue_name', 'MOL')
            atom.SetIntProp('residue_idx', 1)
            atom.SetProp('chain_idx', 'A')
            internal_atom_idx += 1

        atom.SetDoubleProp('charge', atom.GetDoubleProp('_GasteigerCharge'))
        atom.SetDoubleProp('x', atom_positions[atom_idx, 0])
        atom.SetDoubleProp('y', atom_positions[atom_idx, 1])
        atom.SetDoubleProp('z', atom_positions[atom_idx, 2])


class AtomType:
    def __init__(self):
        self.atom_type_definition_list = ATOM_TYPE_DEFINITION_LIST

    def assign_atom_types(self, mol: Chem.Mol):
        for atom_type_dict in self.atom_type_definition_list:
            smarts = atom_type_dict['smarts']
            atom_type = atom_type_dict['atype']

            pattern_mol = Chem.MolFromSmarts(smarts)
            pattern_matches = mol.GetSubstructMatches(pattern_mol)
            for pattern_match in pattern_matches:
                atom = mol.GetAtomWithIdx(pattern_match[0])
                atom.SetProp('atom_type', atom_type)

    def get_docking_atom_types(self, mol: Chem.Mol) -> Dict[int, str]:
        atom_ind_type_map = dict()
        for atom_type_dict in self.atom_type_definition_list:
            smarts = atom_type_dict["smarts"]
            atom_type = atom_type_dict["atype"]
            pattern_mol = Chem.MolFromSmarts(smarts)
            pattern_matches = mol.GetSubstructMatches(pattern_mol)
            for pattern_match in pattern_matches:
                atom_ind = pattern_match[0]
                atom_ind_type_map[atom_ind] = atom_type
        return atom_ind_type_map


class RotatableBond:
    def __init__(self,
                 min_macrocycle_size: int = 7,
                 max_macrocycle_size: int = 33,
                 double_bond_penalty: int = 50,
                 max_breaks: int = 4):

        self.rotatable_bond_smarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        self.amide_bond_smarts = '[C&$(C=O)]-[N&$(NC=O);v3;H1,H2;0]'
        self.conjugate_bond_smarts = '*=*[*]=,#,:[*]'
        self.rotatable_bond_pattern = Chem.MolFromSmarts(self.rotatable_bond_smarts)
        self.amide_bond_pattern = Chem.MolFromSmarts(self.amide_bond_smarts)
        self.conjugate_bond_pattern = Chem.MolFromSmarts(self.conjugate_bond_smarts)

        self.min_macrocycle_size = min_macrocycle_size
        self.max_macrocycle_size = max_macrocycle_size
        self.double_bond_penalty = double_bond_penalty
        self.max_breaks = max_breaks

    def identify_rotatable_bonds(self, mol: Chem.Mol):
        default_rotatable_bond_info_list = list(mol.GetSubstructMatches(self.rotatable_bond_pattern))
        amide_rotatable_bond_info_list = list(mol.GetSubstructMatches(self.amide_bond_pattern))

        for amide_rotatable_bond_info in amide_rotatable_bond_info_list:
            amide_rotatable_bond_info_reversed = tuple(reversed(amide_rotatable_bond_info))
            if amide_rotatable_bond_info in default_rotatable_bond_info_list:
                default_rotatable_bond_info_list.remove(amide_rotatable_bond_info)
            elif amide_rotatable_bond_info_reversed in default_rotatable_bond_info_list:
                default_rotatable_bond_info_list.remove(amide_rotatable_bond_info_reversed)

        return default_rotatable_bond_info_list


class TopologyBuilder:
    def __init__(self, mol: Chem.Mol):
        self.mol = mol

        self.atom_typer = AtomType()
        self.rotatable_bond_finder = RotatableBond()

    def build_molecular_graph(self):
        mol = self.mol

        self.atom_typer.assign_atom_types(mol)
        ComputeGasteigerCharges(mol)
        assign_atom_properties(mol)
        rotatable_bond_info_list = self.rotatable_bond_finder.identify_rotatable_bonds(mol)

        bond_list = list(mol.GetBonds())
        rotatable_bond_idx_list = []
        for bond in bond_list:
            bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            if bond_info in rotatable_bond_info_list or bond_info_reversed in rotatable_bond_info_list:
                rotatable_bond_idx_list.append(bond.GetIdx())

        if len(rotatable_bond_idx_list) != 0:
            splitted_mol = FragmentOnBonds(mol, rotatable_bond_idx_list, addDummies=False)
            splitted_mol_list = list(GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False))
        else:
            splitted_mol_list = [mol]

        num_fragments = len(splitted_mol_list)

        # 1. Find fragment as the root node
        num_fragment_atoms_list = [0] * num_fragments
        for fragment_idx in range(num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            num_atoms = fragment.GetNumAtoms()
            num_fragment_atoms_list[fragment_idx] = num_atoms

        root_fragment_idx = np.argmax(num_fragment_atoms_list)

        # 2. Build torsion tree

        # 2-1. Add atom info into nodes
        torsion_tree = nx.Graph()
        node_idx = 0
        root_fragment = splitted_mol_list[root_fragment_idx]
        num_root_atoms = root_fragment.GetNumAtoms()
        atom_info_list = [dict()] * num_root_atoms
        for root_atom_idx in range(num_root_atoms):
            root_atom = root_fragment.GetAtomWithIdx(root_atom_idx)
            atom_info_dict = {}
            atom_info_dict['sdf_atom_idx'] = root_atom.GetIntProp('sdf_atom_idx')
            atom_info_dict['atom_name'] = root_atom.GetProp('atom_name')
            atom_info_dict['residue_name'] = root_atom.GetProp('residue_name')
            atom_info_dict['chain_idx'] = root_atom.GetProp('chain_idx')
            atom_info_dict['residue_idx'] = root_atom.GetIntProp('residue_idx')
            atom_info_dict['x'] = root_atom.GetDoubleProp('x')
            atom_info_dict['y'] = root_atom.GetDoubleProp('y')
            atom_info_dict['z'] = root_atom.GetDoubleProp('z')
            atom_info_dict['charge'] = root_atom.GetDoubleProp('charge')
            atom_info_dict['atom_type'] = root_atom.GetProp('atom_type')

            atom_info_list[root_atom_idx] = atom_info_dict

        torsion_tree.add_node(node_idx, atom_info_list=atom_info_list)
        node_idx += 1

        for fragment_idx in range(num_fragments):
            if fragment_idx == root_fragment_idx:
                continue
            else:
                fragment = splitted_mol_list[fragment_idx]
                num_fragment_atoms = fragment.GetNumAtoms()
                atom_info_list = [dict()] * num_fragment_atoms
                for atom_idx in range(num_fragment_atoms):
                    atom = fragment.GetAtomWithIdx(atom_idx)
                    atom_info_dict = {}
                    atom_info_dict['sdf_atom_idx'] = atom.GetIntProp('sdf_atom_idx')
                    atom_info_dict['atom_name'] = atom.GetProp('atom_name')
                    atom_info_dict['residue_name'] = atom.GetProp('residue_name')
                    atom_info_dict['chain_idx'] = atom.GetProp('chain_idx')
                    atom_info_dict['residue_idx'] = atom.GetIntProp('residue_idx')
                    atom_info_dict['x'] = atom.GetDoubleProp('x')
                    atom_info_dict['y'] = atom.GetDoubleProp('y')
                    atom_info_dict['z'] = atom.GetDoubleProp('z')
                    atom_info_dict['charge'] = atom.GetDoubleProp('charge')
                    atom_info_dict['atom_type'] = atom.GetProp('atom_type')

                    atom_info_list[atom_idx] = atom_info_dict

                torsion_tree.add_node(node_idx, atom_info_list=atom_info_list)
                node_idx += 1

        # 2-2. Add edge info
        num_rotatable_bonds = len(rotatable_bond_info_list)
        for edge_idx in range(num_rotatable_bonds):
            rotatable_bond_info = rotatable_bond_info_list[edge_idx]
            begin_atom_idx = rotatable_bond_info[0]
            end_atom_idx = rotatable_bond_info[1]

            begin_atom = mol.GetAtomWithIdx(begin_atom_idx)
            begin_sdf_atom_idx = begin_atom.GetIntProp('sdf_atom_idx')
            begin_atom_name = begin_atom.GetProp('atom_name')

            end_atom = mol.GetAtomWithIdx(end_atom_idx)
            end_sdf_atom_idx = end_atom.GetIntProp('sdf_atom_idx')
            end_atom_name = end_atom.GetProp('atom_name')

            begin_node_idx = None
            end_node_idx = None
            for node_idx in range(num_fragments):
                atom_info_list = torsion_tree.nodes[node_idx]['atom_info_list']
                for atom_info_dict in atom_info_list:
                    if atom_info_dict['atom_name'] == begin_atom_name:
                        begin_node_idx = node_idx
                        break
                    elif atom_info_dict['atom_name'] == end_atom_name:
                        end_node_idx = node_idx
                        break

                if begin_node_idx is not None and end_node_idx is not None:
                    break

            if begin_node_idx is None or end_node_idx is None:
                raise ValueError('Bugs in edge assignment code!!')

            torsion_tree.add_edge(begin_node_idx,
                                  end_node_idx,
                                  begin_node_idx=begin_node_idx,
                                  end_node_idx=end_node_idx,
                                  begin_sdf_atom_idx=begin_sdf_atom_idx,
                                  end_sdf_atom_idx=end_sdf_atom_idx,
                                  begin_atom_name=begin_atom_name,
                                  end_atom_name=end_atom_name)

        self.torsion_tree = torsion_tree
        self.mol = mol

    def _deep_first_search(self, node_idx):
        if node_idx == 0:
            self.pdbqt_atom_line_list.append('ROOT\n')
            atom_info_list = self.torsion_tree.nodes[node_idx]['atom_info_list']
            for atom_info_dict in atom_info_list:
                atom_name = atom_info_dict['atom_name']
                self.atom_idx_info_mapping_dict[atom_name] = self.pdbqt_atom_idx
                atom_info_tuple = ('ATOM',
                                   self.pdbqt_atom_idx,
                                   atom_info_dict['atom_name'],
                                   atom_info_dict['residue_name'],
                                   atom_info_dict['chain_idx'],
                                   atom_info_dict['residue_idx'],
                                   atom_info_dict['x'],
                                   atom_info_dict['y'],
                                   atom_info_dict['z'],
                                   1.0,
                                   0.0,
                                   atom_info_dict['charge'],
                                   atom_info_dict['atom_type'])

                self.pdbqt_atom_line_list.append(self.pdbqt_atom_line_format.format(*atom_info_tuple))
                self.pdbqt_atom_idx += 1

            self.pdbqt_atom_line_list.append('ENDROOT\n')

        else:
            atom_info_list = self.torsion_tree.nodes[node_idx]['atom_info_list']
            for atom_info_dict in atom_info_list:
                atom_name = atom_info_dict['atom_name']
                if atom_name not in self.atom_idx_info_mapping_dict:
                    self.atom_idx_info_mapping_dict[atom_name] = self.pdbqt_atom_idx

                atom_info_tuple = ('ATOM',
                                   self.pdbqt_atom_idx,
                                   atom_info_dict['atom_name'],
                                   atom_info_dict['residue_name'],
                                   atom_info_dict['chain_idx'],
                                   atom_info_dict['residue_idx'],
                                   atom_info_dict['x'],
                                   atom_info_dict['y'],
                                   atom_info_dict['z'],
                                   1.0,
                                   0.0,
                                   atom_info_dict['charge'],
                                   atom_info_dict['atom_type'])

                self.pdbqt_atom_line_list.append(self.pdbqt_atom_line_format.format(*atom_info_tuple))
                self.pdbqt_atom_idx += 1

        self.visited_node_idx_set.add(node_idx)

        neighbor_node_idx_list = list(self.torsion_tree.neighbors(node_idx))
        for neighbor_node_idx in neighbor_node_idx_list:
            if neighbor_node_idx not in self.visited_node_idx_set:
                temp_pdbqt_atom_idx = self.pdbqt_atom_idx
                atom_info_list = self.torsion_tree.nodes[neighbor_node_idx]['atom_info_list']
                for atom_info_dict in atom_info_list:
                    atom_name = atom_info_dict['atom_name']
                    if atom_name not in self.atom_idx_info_mapping_dict:
                        self.atom_idx_info_mapping_dict[atom_name] = temp_pdbqt_atom_idx
                        temp_pdbqt_atom_idx += 1

                edge_info = self.torsion_tree.edges[(node_idx, neighbor_node_idx)]
                begin_node_idx = edge_info['begin_node_idx']
                end_node_idx = edge_info['end_node_idx']
                begin_atom_name = edge_info['begin_atom_name']
                end_atom_name = edge_info['end_atom_name']

                if begin_node_idx == node_idx:
                    parent_atom_name = begin_atom_name
                    offspring_atom_name = end_atom_name
                else:
                    parent_atom_name = end_atom_name
                    offspring_atom_name = begin_atom_name

                parent_atom_idx = self.atom_idx_info_mapping_dict[parent_atom_name]
                offspring_atom_idx = self.atom_idx_info_mapping_dict[offspring_atom_name]

                self.branch_info_list.append(
                    (parent_atom_name, str(parent_atom_idx), offspring_atom_name, str(offspring_atom_idx)))
                self.pdbqt_atom_line_list.append(
                    self.pdbqt_branch_line_format.format('BRANCH', parent_atom_idx, offspring_atom_idx))
                self._deep_first_search(neighbor_node_idx)
                self.pdbqt_atom_line_list.append(
                    self.pdbqt_end_branch_line_format.format('ENDBRANCH', parent_atom_idx, offspring_atom_idx))

    def write_pdbqt_file(self, out_file: Union[str, bytes, os.PathLike]):
        self.pdbqt_remark_line_list = []
        self.pdbqt_atom_line_list = []

        self.pdbqt_remark_torsion_line_format = '{:6s}   {:^2d}  {:1s}    {:7s} {:6s} {:^7s}  {:3s}  {:^7s}\n'
        self.pdbqt_atom_line_format = '{:4s}  {:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}\n'
        self.pdbqt_branch_line_format = '{:6s} {:3d} {:3d}\n'
        self.pdbqt_end_branch_line_format = '{:9s} {:3d} {:3d}\n'
        self.torsion_dof_line_format = '{:7s} {:d}'

        # 1. Prepare PDBQT atom lines
        self.atom_idx_info_mapping_dict = {}
        self.branch_info_list = []
        self.visited_node_idx_set = set()
        self.pdbqt_atom_idx = 1

        self._deep_first_search(0)
        self.num_torsions = len(self.branch_info_list)
        self.pdbqt_atom_line_list.append(self.torsion_dof_line_format.format('TORSDOF', self.num_torsions))

        # 2. Prepare PDBQT remark lines
        self.pdbqt_remark_line_list.append('REMARK  ' + str(self.num_torsions) + ' active torsions:\n')
        self.pdbqt_remark_line_list.append("REMARK  status: ('A' for Active; 'I' for Inactive)\n")
        for torsion_idx in range(self.num_torsions):
            branch_info_tuple = self.branch_info_list[torsion_idx]
            remark_torsion_info_tuple = ('REMARK',
                                         torsion_idx + 1,
                                         'A',
                                         'between',
                                         'atoms:',
                                         branch_info_tuple[0] + '_' + branch_info_tuple[1],
                                         'and',
                                         branch_info_tuple[2] + '_' + branch_info_tuple[3])

            self.pdbqt_remark_line_list.append(self.pdbqt_remark_torsion_line_format.format(*remark_torsion_info_tuple))

        self.pdbqt_line_list = self.pdbqt_remark_line_list + self.pdbqt_atom_line_list

        with open(out_file, 'w') as f:
            for pdbqt_line in self.pdbqt_line_list:
                f.write(pdbqt_line)

    def write_constraint_bpf_file(self, out_path: Union[str, bytes, os.PathLike]):
        self.core_bpf_remark_line_list = []
        self.core_bpf_atom_line_list = []
        self.core_bpf_atom_line_format = '{:8.3f}\t{:8.3f}\t{:8.3f}\t{:6.2f}\t{:6.2f}\t{:3s}\t{:<2s}\n'

        self.core_bpf_remark_line_list.append('x y z Vset r type atom\n')

        root_atom_info_list = self.torsion_tree.nodes[0]['atom_info_list']
        for atom_info_dict in root_atom_info_list:
            atom_info_tuple = (atom_info_dict['x'],
                               atom_info_dict['y'],
                               atom_info_dict['z'],
                               -1.2,
                               0.6,
                               'map',
                               atom_info_dict['atom_type'])

            self.core_bpf_atom_line_list.append(self.core_bpf_atom_line_format.format(*atom_info_tuple))

        self.core_bpf_line_list = self.core_bpf_remark_line_list + self.core_bpf_atom_line_list

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, 'w') as ligand_core_bpf_file:
            for core_bpf_line in self.core_bpf_line_list:
                ligand_core_bpf_file.write(core_bpf_line)

    def get_sdf_torsion_tree_info(self) -> Tuple[str, str, str, str]:
        frag_info_str = ''
        torsion_info_str = ''
        atom_info_str = ''

        num_nodes = self.torsion_tree.number_of_nodes()
        num_edges = self.torsion_tree.number_of_edges()

        for node_idx in range(num_nodes):
            atom_info_list = self.torsion_tree.nodes[node_idx]['atom_info_list']
            for atom_info_dict in atom_info_list:
                frag_info_str += str(atom_info_dict['sdf_atom_idx'])
                frag_info_str += ' '

            frag_info_str = frag_info_str[:-1]
            frag_info_str += '\n'

        edge_key_list = list(self.torsion_tree.edges.keys())
        for edge_idx in range(num_edges):
            edge_key = edge_key_list[edge_idx]
            edge_info_dict = self.torsion_tree.edges[edge_key]
            begin_sdf_atom_idx = str(edge_info_dict['begin_sdf_atom_idx'])
            end_sdf_atom_idx = str(edge_info_dict['end_sdf_atom_idx'])
            begin_node_idx = str(edge_info_dict['begin_node_idx'])
            end_node_idx = str(edge_info_dict['end_node_idx'])

            torsion_info_str += f'{begin_sdf_atom_idx} {end_sdf_atom_idx} {begin_node_idx} {end_node_idx}'
            torsion_info_str += '\n'

        for atom in self.mol.GetAtoms():
            sdf_atom_idx = str(atom.GetIntProp('sdf_atom_idx'))
            charge = str(atom.GetDoubleProp('charge'))
            if charge == "nan":
                charge = "0"
            atom_type = atom.GetProp('atom_type')
            atom_info = str(sdf_atom_idx).ljust(3) + charge[:10].ljust(10) + atom_type.ljust(2)
            atom_info_str += atom_info
            atom_info_str += '\n'
        
        frag_all_info_str = " ".join([str(i) for i in range(1, 1 + self.mol.GetNumAtoms())])

        return frag_info_str, frag_all_info_str, torsion_info_str, atom_info_str

    def write_sdf_file(self, out_file: Union[str, bytes, os.PathLike], do_rigid_docking: bool = False):
        frag_info_str, frag_all_info_str, torsion_info_str, atom_info_str = self.get_sdf_torsion_tree_info()
        if do_rigid_docking:
            self.mol.SetProp("fragInfo", frag_all_info_str)
        else:
            self.mol.SetProp("fragInfo", frag_info_str)
            self.mol.SetProp("torsionInfo", torsion_info_str)
        self.mol.SetProp("atomInfo", atom_info_str)
        with Chem.SDWriter(str(out_file)) as writer:
            writer.write(self.mol)