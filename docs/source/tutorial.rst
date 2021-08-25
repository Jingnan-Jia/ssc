Tutorial
========

This tutorial is to demonstrate how to use the code for the project **SSc score prediction**.

Position prediction
-------------------
#. Training and validation for 4 folds separately.

    .. code-block:: bash

        python run_pos.py --net="vgg11_3d" --fold=1 --mode='train'
        python run_pos.py --net="vgg11_3d" --fold=2 --mode='train'
        python run_pos.py --net="vgg11_3d" --fold=3 --mode='train'
        python run_pos.py --net="vgg11_3d" --fold=4 --mode='train'

    Or using slurm scripts (put the above 4 lines into `script_pos.sh` following some configures):

    .. code-block:: bash

        sbatch script_pos.sh

#. Merge the prediction of 4 folds and calculate the metrics based on the whole dataset:



Score prediction
-------------------
#. Training and validation for 4 folds separately:

    .. code-block:: bash

        python run.py --net="vgg11_bn" --fold=1 --mode='train'
        python run.py --net="vgg11_bn" --fold=2 --mode='train'
        python run.py --net="vgg11_bn" --fold=3 --mode='train'
        python run.py --net="vgg11_bn" --fold=4 --mode='train'

    Or using slurm scripts (put the above 4 lines into `script.sh` following some configures):

    .. code-block:: bash

        sbatch script.sh

#. Merge the prediction of 4 folds and calculate the metrics based on the whole dataset:

Inference by cascaded networks
--------------------------------


Train another network to refine position prediction
--------------------------------------------------------


Knowledge distillation for 3D network
-------------------------------------

Tune hyper-parameters
-----------------------

Common in `run` and `run_pos`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following arguments are from :mod:`ssc_scoring.mymodules.set_args` and :mod:`ssc_scoring.mymodules.set_args_pos`.

#. `--mode`

    Mode includes 'train', 'infer', 'continue_train', 'transfer_learning'.
    'train' means training a network.
    'infer' means evaluate a trained network. In this mode, `--eval_id` need to be specified.
    'continue_train` means continue training based on pre-trained weights. In this mode, `--eval_id` need to be specified.
    `transfer_learning` means initiate the encoder part of a network, and train the whole network.
    .. code-block:: bash

            python run.py --net="vgg11_bn" --fold=1 --mode='valid' --eval_id=193



#. `--eval_id`

    Evaluate trained networks. If the experiment ID of the trained network is 193,

    .. code-block:: bash

            python run.py --net="vgg11_bn" --fold=1 --mode='valid' --eval_id=193


#. `--net`

    Use different net structure.

    .. code-block:: bash

            python run.py --net="cnn3fc1" --fold=1 --mode='train'


#. `--fc1_nodes`, `--fc2_nodes`

    Set the node number of fully connected layer.

    .. code-block:: bash

            python run.py --net="vgg16" --fold=1 --mode='train' --fc1_nodes=256 --fc1_nodes=128


#. `--total_folds`, `--fold`

    Set the total folds and fold number.
    .. code-block:: bash

            python run.py --total_folds=4 --fold=1
            python run.py --total_folds=4 --fold=2
            python run.py --total_folds=4 --fold=3
            python run.py --total_folds=4 --fold=4


#. `--valid_period`

    How many epochs between 2 validation steps during training.

    .. code-block:: bash

        python run.py --mode='train' --valid_period=5


#. `--workers`

    Number of workers for dataloader (trainloader, validloader and testloader).

    .. code-block:: bash

        python run.py --mode='train' --workers=6


#. `--ts_level_nb`

    If set to 240, the testing labels would include all possible values and have the similar distribution with
    training/validation dataset.
    240 here means 240 2D slices in testing dataset. The 240 slices are from 48 selected 3D CT images.

    .. code-block:: bash

        python run.py --mode='train' --ts_level_nb=240


#. `--loss`

    Loss function name.

    .. code-block:: bash

        python run.py --mode='train' --loss='mse'


#. `--pretrained`

    If using the pretrained weights from ImageNet. This is used if `--mode='train'`. If yoiu want to evaluate
    your trained model, please use `--mode='infer` --eval_id=[ex_id]`.

    It only works for `--net='vgg11_bn', or 'vgg16', 'vgg19', 'resnet18', 'resnext50_32x4d', 'resnext101_32x8d'`.

    .. code-block:: bash

        python run.py --mode='train' --pretrained=1


#. `--epochs`

    The number of training epochs. Normally it is set to 200 to 1000.

    .. code-block:: bash

        python run.py --mode='train' --epochs=1000


#. `--weight_decay`

    Ack as L2 weights regularization. Normally it is set to 1e-4. Its effect could be seen at ......

    .. code-block:: bash

        python run.py --mode='train' --weight_decay=1e-4


#. `--outfile`

    Where to save the output log.

    .. code-block:: bash

        python run.py --mode='train' --outfile=my_log.txt


#. `--hostname`

    Hostname of the server.

    .. code-block:: bash

        python run.py --mode='train' --hostname=$(hostname)


#. `--remark`

    Remark for this experiment.

    .. code-block:: bash

        python run.py --mode='train' --net='vgg19' --remark="train vgg19"



Exclusive in `run`
~~~~~~~~~~~~~~~~~~~~

The following arguments are from :mod:`ssc_scoring.mymodules.set_args`

#. `--level`

    Which level is the training data from?

    - Normally we use data from all levels:

    .. code-block:: bash

            python run.py --level=0

    - Or we can also use data from a specific level:

    .. code-block:: bash

            python run.py --level=3



#. `--sampler`

    If use balanced sampler to make the label distribution balanced.

    .. code-block:: bash

        python run.py --sampler=1


#. `--corse_pred_id`

    todo

    .. code-block:: bash

            python run.py --net="vgg16" --fold=1 --mode='train' --fc1_nodes=256 --fc1_nodes=128


#. `--sys`, `sys_ratio`, `sys_pro_in_0`

    Synthetic data setting. `--sys` denotes if using synthetic data; `sys_ratio` denotes the ratio of synthetic data in
    the whole dataset; `sys_pro_in_0` denotes

    .. code-block:: bash

        python run.py --sampler=1


#. `--masked_by_lung`

    If the input ct images are masked by lung area.

    .. code-block:: bash

        python run.py --mode='train' --masked_by_lung=1


#. `--gg_increase`


    Increase the pixel values of synthetic ground glass area when using `blur` method to simulate GG pattern.
    `gg_increase` is a float number to represent how much the pixel-values' increase.
    Because the whole pixel values are truncated to -1500 to 1500.

    .. warning::
        Need to be checked if the description is correct.

    .. code-block:: bash

        python run.py --mode='train' --gg_increase=0.1


#. `--retp_blur`, `--gg_blur`

    How many pixels are used as the smoothed edge between synthetic pattern and healthy images.

    .. code-block:: bash

        python run.py --mode='train' --retp_blur=20 --gg_blur=20


#. `--gen_gg_as_retp`

    How many pixels are used as the smoothed edge between synthetic pattern and healthy images.

    .. code-block:: bash

        python run.py --mode='train' --gen_gg_as_retp=1




Exclusive in `run_pos`
~~~~~~~~~~~~~~~~~~~~~~~

The following arguments are from :mod:`ssc_scoring.mymodules.set_args_pos`.

#. `--train_on_level`, `--level_node`

    `level_node` is specified when your network has extra input node for level information apart the normal input
        node for images.

    `train_on_level` is switched on when you want your network to output only one level. Then the transform will
     crop a 3D region in which this level must be visible.

    .. code-block:: bash

            python run.py --train_on_level=0 --level_node=0

            python run.py --train_on_level=0 --level_node=1

            python run.py --train_on_level=1 --level_node=0
            python run.py --train_on_level=2 --level_node=0
            python run.py --train_on_level=3 --level_node=0
            python run.py --train_on_level=4 --level_node=0
            python run.py --train_on_level=5 --level_node=0


#. `--kd`, `--kd_t_name`

    todo:

    .. code-block:: bash

            python run.py


#. `--infer_2nd`

    todo:

    .. code-block:: bash

            python run.py


#. `--resample_z`

    Resampled image size.

    .. code-block:: bash

            python run.py


#. `--z_size`, `--y_size`, `--x_size`

    Patch size.

    .. code-block:: bash

            python run.py --resample_z=256 --z_size=192 --y_size=256 --x_size=256


#. `--batch_size`

    Batch size.

    .. code-block:: bash

            python run.py --batch_size=4


#. `--infer_stride`

    Stride during inference. Smaller stride lead to better results but require more time.

    .. code-block:: bash

            python run.py --mode='infer' --infer_stride=4


FAQ
~~~~~
#. Q: Difference between `--mode='continue_train' --eval_id=193` and `--pretrained`?
    A:  `--pretrained` means to initiate network by the publich released weights trained from ImageNet. While
    `--mode='continue_train' --eval_id=193` means to initiate network by our previous trained weights trained from our own dataset.
    `--mode='continue_train'` will overwrite `--pretrained`.

#. Q: batch_size disappeared in set_args.py
    A: todo
