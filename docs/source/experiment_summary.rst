Experiment summary
==================

Status
-----------------------------------------------------------------------
[position prediction]: Latest summary **NOT** done.
[score prediction]: Latest summary done.







Performance of refined PosNet (train high resolution CT)
-----------------------------------------------------------------------

[position prediction]
**TLDR: todo.**

- Directly input high resolution CT patches (192 * 512 * 512) into network.
    **TLDR** Bad effect.**

    This patch can only cover one or two levels. So the labels (relative slice number) for those levels outside
    the patch would be 0 (lower bound) or 192 (upper bound). To get the position prediction, during inference, we need
    to have a slice window to output a lot of positions and do some post processing. This method was inspired by the
    following reference.

    References:
        S. Belharbi et al., “Spotting L3 slice in CT scans using deep convolutional network and transfer learning,”
        Comput. Biol. Med., vol. 87, pp. 95–103, Aug. 2017, doi: 10.1016/j.compbiomed.2017.05.018.

    The trained results are pretty bad. It does not have clear peak point as shown in the above reference. So I did not
    try it anymore.


- With `train_on_level` to receive patches from a region which include a specific level position, and output this level.

    Compared with the upper method, current method has 2 differences: 1) the input patches are not cropped from the
    whole image, but from the region which include a specified level position. 2） the labels only include one level.

    Therefore, we have to train 5 different networks for 5 different level position prediction.

    ==============  ========    ========    ========    ========    =========
    train_on_level  Fold1        Fold2       Fold3       Fold4      ave_MAE
    ==============  ========    ========    ========    ========    =========
    1               602->462    601->463    603->459    604->457    3.8925
    2               600->464    599->461    688->466    ->465
    3               596->468    595->467    598->469    597->470    6.2325
    4               571->528    572->529    573->520    574->519    2.84
    5               589->530    686->532    590->512    687->527    2.6425
    Average                                                         3?
    ==============  ========    ========    ========    ========    =========


- With `level_node` to use an extra node to receive an extra input (level information) in fully connected layer. So the number of nodes in
fully connected layer became 1025 from 1024.

    **TLDR** Bad effect.**

    The network did not coverage. I think this is because that the extra node is ignored by the
    network. 1 node in 1025 nodes is really small.

    `Level_node=5`:

    ==============  ====    ====    ====    ====    ====
    Fold            1       2       3       4       ave
    ==============  ====    ====    ====    ====    ====
    Experiment ID   455     456     460     458     ??
    ==============  ====    ====    ====    ====    ====

    .. Note::
        Do not use extra node. Directory multiply the level information by fully connected layer.


Performance of cascaded refined PosNet
-----------------------------------------------------------------------

[PosNet_1 + PosNet_2]
**TLDR: todo.**


Performance of cascaded networks
-----------------------------------------------------------------------

[PosNet + ScoreNet]
**TLDR: Very good! The PosNet can not affect the final results**. More details please see word documents.

PosNet: 193_194_276_277
ScoreNet: 1405_1404_1411_1410

Pure ScoreNet performance:

==================  =====
valid_WK_disext     0.647
valid_WK_gg


PosNet + ScoreNet performance:

==================  =====
valid_WK_disext     0.644
valid_WK_gg

Effect of different normalization methods
--------------------------------------------------------------------

[score prediction]
**TLDR: no improvement.**

Different normalization methods could be achieved placing one of the following 3 transforms could be placed at the end
of :meth:`ssc_scoring.mymodules.composed_trans.xformd_score`.

- `NormImgPosd()`: standard normalization. Mean=0, std=1.
- `NormNeg1To1d()`: rescaled to [-1, 1].
- `RescaleToNeg1500Pos1500d()`: rescaled to [-1500, 1500].
- Without any normalization.

I used `NormImgPosd()` for the previous experiments. After the discussion with Anna, we noticed that the scores are
sometimes obtained by compare the slice with other slices. So some looked healthy slices have lower socres. So Berend
adviced to use the original pixel values to make sure the pixel values. So I removed the normalization layer and also
tried the other 3 methods.

- The ex ID for `RescaleToNeg1500Pos1500d()`, `--gg_as_ret=1`: 1105, 1106, 1107, 1108
    valid_mae_end5 = 5.04 (averaged in 4 folds) [**Baseline**]

- The ex ID for `RescaleToNeg1500Pos1500d()`, `--gg_as_ret=1`: 1605, 1606, 1608, 1607
    valid_mae_end5 = 5.062 (averaged in 4 folds)

- The ex ID for `RescaleToNeg1500Pos1500d()`, `--gg_as_ret=0`: 1595, 1596, 1599, 1597
    valid_mae_end5 = 5.08 (averaged in 4 folds)

- The ex ID for `NormNeg1To1d()`: Not started
    valid_mae_end5 = None (averaged in 4 folds)

- The ex ID for `Without any Norm`, `--gg_as_ret=1`: 1585, 1586, 1587, 1588
    valid_mae_end5 = 5.115 (averaged in 4 folds)



Using more patches as the seed of synthetic data
-----------------------------------------------------------------------
[score prediction]

**TLDR: No effect**.

experiments' ID (vgg11_3d): 1658, 1659, 1657, 1656. valid_mae_end5 = 4.99 (averaged in 4 folds)

experiments' ID (resnet18): 1660, 1661, 1663, 1662. valid_mae_end5 = 5.2575 (averaged in 4 folds)


.. warning::
    **The following experiments are based on wrong code! Because only a random patch is selected as the seed instead of
    all of the patches.**

    - Using more patches as the seed of synthetic data:

        **TLDR: Bad effect**. valid_mae_end5 = 5.0575 (averaged in 4 folds)

        In the previous experiments, all synthetic RETP patterns are from the same patch, similarly, all synthetic GG patterns
        are also from the same patch. We hope to obtain more patches which are full of the two patterns as the seed to generate
        more samples. These patches were carefully cropped by Jingnan in advance. By this way, I expected to see better results.

        experiments' ID: 1614, 1615, 1612, 1613


    - Using more patches as the seed of synthetic data using bigger net

        **TLDR: Bad effect**. valid_mae_end5 = 5.37 (averaged in 4 folds)

        I thought maybe bigger net can benefit from more variable syntheic data. So I trained Resnet18.

        experiments' ID: 1617, 1616, 1618, 1619



[position prediction]
**TLDR: todo.**

Performance of KD
-----------------------------------------------------------------------

[position prediction]
**TLDR: todo.**
