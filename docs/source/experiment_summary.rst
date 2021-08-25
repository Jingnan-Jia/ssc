Experiment summary
==================


















Performance of refined PosNet
-----------------------------------------------------------------------

[position prediction]
**TLDR: todo.**


Performance of cascaded refined PosNet
-----------------------------------------------------------------------

[PosNet_1 + PosNet_2]
**TLDR: todo.**


Performance of cascaded networks
-----------------------------------------------------------------------

[PosNet + ScoreNet]
**TLDR: todo.**


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

The ex ID for `RescaleToNeg1500Pos1500d()`, `--gg_as_ret=1`: 1105, 1106, 1107, 1108 [**Baseline**]
valid_mae_end5 = 5.04 (averaged in 4 folds)

The ex ID for `RescaleToNeg1500Pos1500d()`, `--gg_as_ret=1`: 1605, 1606, 1608, 1607
valid_mae_end5 = 5.062 (averaged in 4 folds)

The ex ID for `RescaleToNeg1500Pos1500d()`, `--gg_as_ret=0`: 1595, 1596, 1599, 1597
valid_mae_end5 = 5.08 (averaged in 4 folds)

The ex ID for `NormNeg1To1d()`: Not started
valid_mae_end5 = None (averaged in 4 folds)

The ex ID for `Without any Norm`, `--gg_as_ret=1`: 1585, 1586, 1587, 1588
valid_mae_end5 = 5.115 (averaged in 4 folds)


Several issures:

#. where is the ex id without any normalization?


Using more patches as the seed of synthetic data
-----------------------------------------------------------------------
[score prediction]
**TLDR: No effect**. valid_mae_end5 = 5.0575 (averaged in 4 folds)

In the previous experiments, all synthetic RETP patterns are from the same patch, similarly, all synthetic GG patterns
are also from the same patch. We hope to obtain more patches which are full of the two patterns as the seed to generate
more samples. These patches were carefully cropped by Jingnan in advance. By this way, I expected to see better results.

experiments' ID: 1614, 1615, 1612, 1613


Using more patches as the seed of synthetic data using bigger net
-----------------------------------------------------------------------
[score prediction]
**TLDR: Bad effect**. valid_mae_end5 = 5.37 (averaged in 4 folds)

I thought maybe bigger net can benefit from more variable syntheic data. So I trained Resnet18.

experiments' ID: 1617, 1616, 1618, 1619



[position prediction]
**TLDR: todo.**

Performance of KD
-----------------------------------------------------------------------

[position prediction]
**TLDR: todo.**
