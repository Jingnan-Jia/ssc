Experiment summary
==================






















Effect of different normalization methods [score prediction].
--------------------------------------------------------------------

**TLDR: no effect.**

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


