import os
import torch
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn
import numpy as np
from ase import Atoms
from torch.xpu import device

if __name__ == '__main__':
    qm9tut = './qm9tut'
    best_model = torch.load(os.path.join(qm9tut, 'best_inference_model'), map_location='cpu',weights_only=False)

    # qm9data = QM9(
    #     './qm9.db',
    #     batch_size=100,
    #     num_train=1000,
    #     num_val=1000,
    #     transforms=[
    #         trn.ASENeighborList(cutoff=5.),
    #         trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),  ### 预处理，去除能量的偏移量
    #         trn.CastTo32()
    #     ],
    #     property_units={QM9.U0: 'eV'},
    #     num_workers=1,
    #     split_file=os.path.join(qm9tut, "split.npz"),
    #     pin_memory=True,  # set to false, when not using a GPU
    #     load_properties=[QM9.U0],  # only load U0 property
    # )
    # qm9data.prepare_data()
    # qm9data.setup()
    #
    #
    # for batch in qm9data.test_dataloader():
    #     result = best_model(batch)
    #     print("Result dictionary:", result)
    #     break

    converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)

    numbers = np.array([6, 1, 1, 1, 1])
    positions = np.array([[-0.0126981359, 1.0858041578, 0.0080009958],
                          [0.002150416, -0.0060313176, 0.0019761204],
                          [1.0117308433, 1.4637511618, 0.0002765748],
                          [-0.540815069, 1.4475266138, -0.8766437152],
                          [-0.5238136345, 1.4379326443, 0.9063972942]])
    atoms = Atoms(numbers=numbers, positions=positions)

    # inputs = converter(atoms)
    #
    # print('Keys:', list(inputs.keys()))
    #
    # pred = best_model(inputs)
    #
    # print('Prediction:', pred[QM9.U0])

    calculator = spk.interfaces.SpkCalculator(
        model_file=os.path.join(qm9tut, "best_inference_model"),  # path to model
        neighbor_list=trn.ASENeighborList(cutoff=5.),  # neighbor list
        energy_key=QM9.U0,  # name of energy property in model
        energy_unit="eV",  # units of energy property
        device=torch.device('cuda:0'),  # device for computation
    )
    atoms.set_calculator(calculator)
    print('Prediction:', atoms.get_total_energy())