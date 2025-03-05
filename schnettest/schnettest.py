from schnetpack.datasets import QM9
from schnetpack.transform import ASENeighborList
if __name__ == '__main__':

    qm9data = QM9(
        './qm9.db',
        batch_size=10,
        num_train=110000,
        num_val=10000,
        split_file='split_qm9.npz',
        num_workers=0,
        num_val_workers=0,
        transforms=[ASENeighborList(cutoff=5.)]
    )

    qm9data.prepare_data()
    qm9data.setup()



    print('Number of reference calculations:', len(qm9data.dataset))
    print('=================================')
    print('Number of train data:', len(qm9data.train_dataset))
    print('=================================')
    print('Number of validation data:', len(qm9data.val_dataset))
    print('=================================')
    print('Number of test data:', len(qm9data.test_dataset))
    print('=================================')
    print('Available properties:')
    print('=================================')

    for p in qm9data.dataset.available_properties:
        print('-', p)




    print('=================================')
    example = qm9data.dataset[0]
    print('Properties:')

    for k, v in example.items():
        print('-', k, ':', v.shape)

    """
    - _idx : torch.Size([1])
    - rotational_constant_A : torch.Size([1])       A轴的旋转常数，通常以MHz为单位。它是一个与分子转动惯量相关的物理量，用于描述分子在给定轴上的旋转能级。
    - rotational_constant_B : torch.Size([1])       B轴的旋转常数，同样以MHz为单位。这是另一个与分子转动惯量相关的物理量，对应于分子在另一轴上的旋转能级。
    - rotational_constant_C : torch.Size([1])       C轴的旋转常数，也是以MHz为单位。它是第三个与分子转动惯量相关的物理量，描述分子在第三个轴上的旋转能级。
    - dipole_moment : torch.Size([1])               电偶极矩，通常以Debye（D）为单位。它是一个矢量，表示分子中正负电荷中心之间的距离和电荷分布的不对称性。
    - isotropic_polarizability : torch.Size([1])    等向性极化率，通常以Angstrom^3为单位。它描述了分子在外电场作用下电子云变形的能力，是一个标量。
    - homo : torch.Size([1])                        最高占据分子轨道（HOMO）的能量，通常以hartree为单位。这是分子中电子占据的最高能级。
    - lumo : torch.Size([1])                        最低未占据分子轨道（LUMO）的能量，通常以hartree为单位。这是分子中电子未占据的最低能级。
    - gap : torch.Size([1])                         HOMO和LUMO之间的能隙，通常以hartree为单位。这个能隙与分子的化学活性和导电性有关。
    - electronic_spatial_extent : torch.Size([1])   电子空间扩展，这是一个描述分子中电子分布的空间范围的量。
    - zpve : torch.Size([1])                        零点振动能（Zero-Point Vibrational Energy），通常以hartree为单位。这是分子在绝对零度时的振动能量。
    - energy_U0 : torch.Size([1])                   分子的总能量在0K时的值，通常以hartree为单位。
    - energy_U : torch.Size([1])                    分子的总能量，通常以hartree为单位。这个值可能会在不同的温度下计算。
    - enthalpy_H : torch.Size([1])                  分子的焓，通常以hartree为单位。焓是系统的内能加上其体积与外界压力的乘积。
    - free_energy : torch.Size([1])                 分子的自由能，通常以hartree为单位。自由能是系统在恒温恒压下进行可逆过程所能做的最大非体积功。
    - heat_capacity : torch.Size([1])               热容，通常以cal/(mol·K)为单位。它描述了系统温度升高1K所需的热量。
    - _n_atoms : torch.Size([1])
    - _atomic_numbers : torch.Size([4])
    - _positions : torch.Size([4, 3])
    - _cell : torch.Size([1, 3, 3])
    - _pbc : torch.Size([3])
    """
    print('=================================')
    print('迭代数据集分区')
    print('val_dataloader_len:', len(qm9data.val_dataloader()))
    i = 0
    for batch in qm9data.val_dataloader():
        print(batch.keys())
        print('System index:', batch['_idx_m'])
        print('Center atom index:', batch['_idx_i'])
        print('Neighbor atom index:', batch['_idx_j'])
        print('Total energy at 0K:', batch[QM9.U0])
        print('HOMO:', batch[QM9.homo])
        if i>1:
            break
        i = i+1
