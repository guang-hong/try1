from rdkit import Chem
from rdkit.Chem import Draw
# import matplotlib.pyplot as plt

# 读取 SDF 文件
supplier = Chem.SDMolSupplier(r'data/raw/pretrain_mols.sdf')

# 提取第一个分子
for mol in supplier:
    mol.GetNumBonds()
    mol.GetConformer()
mol = supplier[0]

# 绘制分子结构
img = Draw.MolToImage(mol, size=(300, 300))
# plt.imshow(img)
# plt.axis('off')
# plt.show()
