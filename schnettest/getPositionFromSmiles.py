from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd


def getAtomsPositions( smiles:str):




    # # 初始化 SMILES 字符串
    # smiles = 'O=N([O-])C1=C(CN=C1NCCSCc2ncccc2)Cc3ccccc3'
    # smiles = 'CCC(C)C(N)C1=NC(CS1)C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](CCC(O)=O)C(=O)N[C@@H]([C@@H](C)CC)C(=O)NCCCC[C@@H]2NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](CC(O)=O)NC(=O)[C@H](Cc3[nH]cnc3)NC(=O)[C@@H](Cc4ccccc4)NC(=O)[C@@H](NC(=O)[C@@H](CCCN)NC2=O)[C@@H](C)CC'
    smiles = 'CC(C)[C@@H]1NC(=O)[C@H](C)OC(=O)C(NC(=O)[C@H](OC(=O)[C@@H](NC(=O)[C@H](C)OC(=O)[C@H](NC(=O)[C@H](OC(=O)[C@@H](NC(=O)[C@H](C)OC(=O)[C@H](NC(=O)[C@H](OC1=O)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C'
    # # 从 SMILES 字符串创建分子
    print(f'========={smiles}====================')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    # 向分子中添加氢原子
    mol3d = Chem.AddHs(mol)

    # 使用 EmbedMolecule 函数嵌入分子以生成 3D 坐标
    # Chem.AllChem.Compute2DCoords(mol3d)
    AllChem.EmbedMolecule(mol3d, useRandomCoords=True)
    if mol3d.GetNumConformers() == 0:
        print("分子没有构象。")
    else:
        print(f"分子有 {mol.GetNumConformers()} 个构象。")
    # 使用 MMFF94 力场进行优化
    AllChem.MMFFOptimizeMolecule(mol3d)

    # 初始化一个字典来存储原子的坐标和原子类型
    atom_positions = {}

    # 遍历分子中的所有原子
    for atom in mol3d.GetAtoms():
        # 获取原子的坐标
        pos = mol3d.GetConformer().GetAtomPosition(atom.GetIdx())
        # 获取原子的符号（类型）
        atom_symbol = atom.GetSymbol()
        # 获取原子的周期表序号
        atomic_number = atom.GetAtomicNum()
        # 将坐标和原子类型存储在字典中
        atom_positions[atom.GetIdx()] = (atom_symbol,atomic_number, pos.x, pos.y, pos.z)

    # 打印原子的坐标和类型
    for idx, (atom_symbol, atomic_number, x, y, z) in atom_positions.items():
        print(f"Atom {idx}: {atom_symbol}({atomic_number}) at coordinates ({x}, {y}, {z})")


if __name__ == '__main__':

    # file_path = r'../data/raw/pretrain_mols.sdf'
    # supp = Chem.SDMolSupplier(file_path)

    # for mol in supp:
    #     getAtomsPositions(mol)

    dataset_name = 'bbbp'
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
        getAtomsPositions(smiles)
        break
