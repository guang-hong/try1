import os
import torch
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np


def optimize_conformer( m, algo="MMFF"):
    print("Calculating {} ...".format( Chem.MolToSmiles(m)))

    mol = Chem.AddHs(m) #补氢原子

    if algo == "ETKDG":
        # Landrum et al. DOI: 10.1021/acs.jcim.5b00654
        k = AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        if k != 0:
            return None

    elif algo == "UFF":
        # Universal Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5)
        try:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except ValueError:
            return None

        if not arr:
            return None

        else:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=20000)
            idx = np.argmin(arr, axis=0)[1]
            conf = mol.GetConformers()[idx]
            mol.RemoveAllConformers()
            mol.AddConformer(conf)

    elif algo == "MMFF":
        # Merck Molecular Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5) #生成50个mol分子的三维构象，并排除rms小于0.5的值
        try:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except ValueError:
            return None

        if not arr:
            return None

        else:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=20000)
            # idx = int(np.argmin(arr, axis=0)[1])
            # conf = mol.GetConformers()[idx]
            # mol.RemoveAllConformers()
            # mol.AddConformer(conf)

    # mol = Chem.RemoveHs(mol)
    return mol


def getAtomsPositions( smiles:str):




    # # 初始化 SMILES 字符串
    # smiles = 'O=N([O-])C1=C(CN=C1NCCSCc2ncccc2)Cc3ccccc3'
    # smiles = 'CCC(C)C(N)C1=NC(CS1)C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](CCC(O)=O)C(=O)N[C@@H]([C@@H](C)CC)C(=O)NCCCC[C@@H]2NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](CC(O)=O)NC(=O)[C@H](Cc3[nH]cnc3)NC(=O)[C@@H](Cc4ccccc4)NC(=O)[C@@H](NC(=O)[C@@H](CCCN)NC2=O)[C@@H](C)CC'
    # smiles = 'CC(C)[C@@H]1NC(=O)[C@H](C)OC(=O)C(NC(=O)[C@H](OC(=O)[C@@H](NC(=O)[C@H](C)OC(=O)[C@H](NC(=O)[C@H](OC(=O)[C@@H](NC(=O)[C@H](C)OC(=O)[C@H](NC(=O)[C@H](OC1=O)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C'
    # smiles = '[H]N1C(=O)[C@@]([H])(C([H])(C([H])([H])[H])C([H])([H])[H])OC(=O)[C@]([H])(C([H])(C([H])([H])[H])C([H])([H])[H])N([H])C(=O)[C@]([H])(C([H])([H])[H])OC(=O)[C@@]([H])(C([H])(C([H])([H])[H])C([H])([H])[H])N([H])C(=O)[C@@]([H])(C([H])(C([H])([H])[H])C([H])([H])'
    # # 从 SMILES 字符串创建分子
    print(f'========={smiles}====================')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    # 向分子中添加氢原子
    # mol3d = Chem.AddHs(mol)
    mol3d = optimize_conformer(mol)

    # 使用 EmbedMolecule 函数嵌入分子以生成 3D 坐标
    # Chem.AllChem.Compute2DCoords(mol3d)
    # AllChem.EmbedMolecule(mol3d, useRandomCoords=True)
    if mol3d.GetNumConformers() == 0:
        print("分子没有构象。")
    else:
        print(f"分子有 {mol.GetNumConformers()} 个构象。")
    # 使用 MMFF94 力场进行优化
    # AllChem.MMFFOptimizeMolecule(mol3d)

    # 初始化一个字典来存储原子的坐标和原子类型
    atom_positions = {}

    numbers = []
    positions = []



    # 遍历分子中的所有原子
    for atom in mol3d.GetAtoms():
        # 获取原子的坐标
        pos = mol3d.GetConformer().GetAtomPosition(atom.GetIdx())
        positions.append(pos)
        # 获取原子的符号（类型）
        atom_symbol = atom.GetSymbol()
        # 获取原子的周期表序号
        atomic_number = atom.GetAtomicNum()
        numbers.append(atom.GetAtomicNum())
        # 将坐标和原子类型存储在字典中
        atom_positions[atom.GetIdx()] = (atom_symbol,atomic_number, pos.x, pos.y, pos.z)

    atoms = Atoms(numbers=numbers, positions=positions)
    # 打印原子的坐标和类型
    for idx, (atom_symbol, atomic_number, x, y, z) in atom_positions.items():
        print(f"Atom {idx}: {atom_symbol}({atomic_number}) at coordinates ({x}, {y}, {z})")

    return atoms


def getAtomsPositionsByMol(mol:Chem.Mol):

    if mol is None:
        return None, None
    # 向分子中添加氢原子
    # mol3d = Chem.AddHs(mol)

    mol3d = optimize_conformer(mol)

    # 使用 EmbedMolecule 函数嵌入分子以生成 3D 坐标
    # Chem.AllChem.Compute2DCoords(mol3d)
    # AllChem.EmbedMolecule(mol3d, useRandomCoords=True)
    if mol3d.GetNumConformers() == 0:
        print("分子没有构象。")
    else:
        print(f"分子有 {mol.GetNumConformers()} 个构象。")
    # 使用 MMFF94 力场进行优化
    AllChem.MMFFOptimizeMolecule(mol3d)

    # 初始化一个字典来存储原子的坐标和原子类型
    atom_positions = {}

    numbers = []
    positions = []



    # 遍历分子中的所有原子
    for atom in mol3d.GetAtoms():
        # 获取原子的坐标
        pos = mol3d.GetConformer().GetAtomPosition(atom.GetIdx())
        positions.append(pos)
        # 获取原子的符号（类型）
        atom_symbol = atom.GetSymbol()
        # 获取原子的周期表序号
        atomic_number = atom.GetAtomicNum()
        numbers.append(atom.GetAtomicNum())
        # 将坐标和原子类型存储在字典中
        atom_positions[atom.GetIdx()] = (atom_symbol,atomic_number, pos.x, pos.y, pos.z)

    atoms = Atoms(numbers=numbers, positions=positions)
    # 打印原子的坐标和类型
    for idx, (atom_symbol, atomic_number, x, y, z) in atom_positions.items():
        print(f"Atom {idx}: {atom_symbol}({atomic_number}) at coordinates ({x}, {y}, {z})")

    return atoms



if __name__ == '__main__':

    qm9tut = './qm9tut'
    best_model = torch.load(os.path.join(qm9tut, 'best_inference_model'), map_location='cpu', weights_only=False)

    dataset_name = 'tox21'
    smiles_column = None

    if dataset_name == 'esol':
        task_names = ['measured log solubility in mols per litre']
        smiles_column = 'smiles'

    if dataset_name == 'freesolv':
        task_names = ['expt']
        smiles_column = 'smiles'

    if dataset_name == 'lipophilicity':
        task_names = ['exp']
        smiles_column = 'smiles'

    if dataset_name == 'tox21':
        smiles_column = 'smiles'

    if dataset_name == 'muv':
        smiles_column = 'smiles'

    if dataset_name == 'clintox':
        smiles_column = 'smiles'

    if dataset_name == 'toxcast':
        smiles_column = 'smiles'

    if dataset_name == 'bace':
        task_names = ['Class']
        smiles_column = 'mol'

    if dataset_name == 'bbbp':
        task_names = ['p_np']
        smiles_column = 'smiles'

    if dataset_name == 'sider':
        smiles_column = 'smiles'

    if dataset_name == 'hiv':
        task_names = ['HIV_active']
        smiles_column = 'smiles'

    df = pd.read_csv('../data/raw/{}.csv'.format(dataset_name))
    smileses = df[smiles_column].tolist()

    for smiles in smileses:
        atoms = getAtomsPositions(smiles)

        converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)


        # inputs = converter(atoms)
        #
        # print('Keys:', list(inputs.keys()))
        #
        # pred = best_model(inputs)

        # print('Prediction:', pred[QM9.U0])

        calculator = spk.interfaces.SpkCalculator(
            model_file=os.path.join(qm9tut, "best_inference_model"),  # path to model
            neighbor_list=trn.ASENeighborList(cutoff=5.),  # neighbor list
            energy_key=QM9.U0,  # name of energy property in model
            energy_unit="eV",  # units of energy property
            device=torch.device('cpu'),  # device for computation
        )
        # atoms.set_calculator(calculator)
        atoms.calc = calculator
        print('Prediction:', atoms.get_total_energy())
        # print('aa', atoms.get_kinetic_energy())
        # print('bb', atoms.get_potential_energy())
        # print('cc', atoms.get_potential_energies())
        break