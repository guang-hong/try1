import schnetpack as spk
import schnetpack.transform as trn

if __name__ == '__main__':

    custom_data = spk.data.AtomsDataModule(
        './new_dataset.db',
        batch_size=10,
        distance_unit='Ang',
        property_units={'energy':'kcal/mol', 'forces':'kcal/mol/Ang'},
        num_train=1000,
        num_val=100,
        transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
            trn.CastTo32()
        ],
        num_workers=1,
        pin_memory=True, # set to false, when not using a GPU
    )
    custom_data.prepare_data()
    custom_data.setup()

    print('Number of reference calculations:', len(custom_data))
    print('Available properties:')

    for p in custom_data.available_properties:
        print('-', p)
    print()

    example = custom_data[0]
    print('Properties of molecule with id 0:')

    for k, v in example.items():
        print('-', k, ':', v.shape)