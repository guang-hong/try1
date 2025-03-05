from ase import Atoms
from schnetpack.data import ASEAtomsData
import numpy as np

if __name__ == '__main__':

    # load atoms from npz file. Here, we only parse the first 10 molecules
    data = np.load('./md17_uracil.npz')

    numbers = data["z"]
    atoms_list = []
    property_list = []
    for positions, energies, forces in zip(data["R"], data["E"], data["F"]):
        ats = Atoms(positions=positions, numbers=numbers)   ###R：坐标，E：能量，F：力
        properties = {'energy': energies, 'forces': forces}
        property_list.append(properties)
        atoms_list.append(ats)


    print('Properties:', property_list[0])

    new_dataset = ASEAtomsData.create(
        './new_dataset.db',
        distance_unit='Ang',  ### 1 埃（Angstrom）等于 10^(−10)米（m）。在处理原子和分子的坐标时，通常使用埃作为单位，因为原子的尺寸大约在这个数量级。
        property_unit_dict={'energy':'kcal/mol', 'forces':'kcal/mol/Ang'}
    )
    new_dataset.add_systems(property_list, atoms_list)

    print('Number of reference calculations:', len(new_dataset))
    print('Available properties:')

    for p in new_dataset.available_properties:
        print('-', p)
    print()
    """
        Number of reference calculations: 133770
        Available properties:
        - energy
        - forces
    """

    example = new_dataset[0]
    print('Properties of molecule with id 0:')

    for k, v in example.items():
        print('-', k, ':', v.shape)
    """
        Properties of molecule with id 0:
        - _idx : torch.Size([1])
        - energy : torch.Size([1])
        - forces : torch.Size([12, 3])
        - _n_atoms : torch.Size([1])
        - _atomic_numbers : torch.Size([12])
        - _positions : torch.Size([12, 3])
        - _cell : torch.Size([1, 3, 3])
        - _pbc : torch.Size([3])
    """
