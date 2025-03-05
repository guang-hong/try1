import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl

if __name__ == '__main__':

    qm9tut = './qm9tut'
    if not os.path.exists('qm9tut'):
        os.makedirs(qm9tut)

    qm9data = QM9(
        './qm9.db',
        batch_size=100,
        num_train=1000,
        num_val=1000,
        transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),  ### 预处理，去除能量的偏移量
            trn.CastTo32()
        ],
        property_units={QM9.U0: 'eV'},
        num_workers=1,
        split_file=os.path.join(qm9tut, "split.npz"),
        pin_memory=True, # set to false, when not using a GPU
        load_properties=[QM9.U0], #only load U0 property
    )
    qm9data.prepare_data()
    qm9data.setup()

    print('===============================')
    atomrefs = qm9data.train_dataset.atomrefs
    print('U0 of hyrogen:', atomrefs[QM9.U0][1].item(), 'eV')  ### 索引与元素周期表序列等价
    print('U0 of carbon:', atomrefs[QM9.U0][6].item(), 'eV')
    print('U0 of oxygen:', atomrefs[QM9.U0][8].item(), 'eV')

    print('===============================')
    means, stddevs = qm9data.get_stats(
        QM9.U0, divide_by_atoms=True, remove_atomref=True
    )
    print('Mean atomization energy / atom:', means.item())
    print('Std. dev. atomization energy / atom:', stddevs.item())


    print('===============设置模型================')
    cutoff = 5.
    n_atom_basis = 30

    pairwise_distance = spk.atomistic.PairwiseDistances()  # calculates pairwise distances between atoms  ### 用于计算原子之间的成对距离
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)  ### 高斯径向基函数，n_rbf 是径向基函数的数量，cutoff 是截断距离，超出这个距离的原子相互作用将被忽略。
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis,  ### n_atom_basis: 原子基函数的数量。
        n_interactions=3,  ### n_interactions: 交互层的数量。
        radial_basis=radial_basis,  ### radial_basis: 之前定义的径向基函数。
        cutoff_fn=spk.nn.CosineCutoff(cutoff)  ### cutoff_fn: 截断函数，这里使用的是余弦截断函数。
    )
    pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=QM9.U0)
      ### Atomwise 是一个输出模块，用于预测每个原子的属性（例如，能量）。
      ### n_in 是输入特征的维度
      ### output_key 是与模型预测相关的特定属性（这里使用的是 QM9 数据集中的 U0，即分子的零点能）。


    nnpot = spk.model.NeuralNetworkPotential(  ### NeuralNetworkPotential 是整个模型，它将 SchNet 表示与输入和输出模块结合起来。
        representation=schnet,
        input_modules=[pairwise_distance],  ### 包含用于处理输入数据的模块
        output_modules=[pred_U0],  ### 包含用于生成预测的模块
        postprocessors=[trn.CastTo64(), trn.AddOffsets(QM9.U0, add_mean=True, add_atomrefs=True)]  ### 是用于后处理的模块，例如将数据转换为64位精度，并添加偏移量以改进模型的预测。
    )

    print('=====================输出模块=====================')
    output_U0 = spk.task.ModelOutput(
        name=QM9.U0,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    print('===================连接模型和训练过程=======================')
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_U0],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
    )

    print('===================训练模型=======================')
    logger = pl.loggers.TensorBoardLogger(save_dir=qm9tut)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(qm9tut, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=qm9tut,
        log_every_n_steps=1,
        max_epochs=3,  # for testing, we restrict the number of epochs
    )
    trainer.fit(task, datamodule=qm9data)