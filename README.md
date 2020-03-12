# DDP 3D U-Net
---

Basic 3D U-Net in pytorch, with support for distriuted data parallel (DDP) training and automatic mixed precision training. Currently configured for N-body -> hydro project.

### Requirements
Developed and tested with:
* pytorch 1.4.0 (via NERSC module pytorch/v1.4.0-gpu)
* [Nvidia apex](https://github.com/NVIDIA/apex) for distributed and mixed precision training 

### Running
Create `expts` directory to store outputs of experiment runs, or alter `train.py` to point to your desired location. For basic single-GPU, single-process training, do
```
python train.py --run_num=00
```
Configurations are currently specified in the `config/UNet.yaml` file. The `default` config will be automatically selected, but you can use the `--config` flag to choose a different config, e.g.:
```
python train.py --run_num=00 --config=some_alternate_config
```

Alternatively, sample SLURM job scripts are provided for single-GPU (`submit1.slr`) and multi-GPU (`submit8.slr`, `submit16.slr`) training.

### Multi-GPU training
Currently, all processes will read from the same HDF5 data file as is required for the N-body -> hydro project. To prevent a deadlock from multiple process reading the same HDF5 file, you must do `export HDF5_USE_FILE_LOCKING=FALSE` before invoking the distributed training.

For muti-process/multi-GPU training, use the `torch.distributed.launch` script:
```
python -m torch.distributed.launch --nproc_per_node=8 train.py --run_num=00 --config=multi8
```

### Misc info
* Saving checkpoints currently does not support mixed precision. Adding this functionality is simple, see the [apex AMP docs](https://nvidia.github.io/apex/amp.html#checkpointing).
* Sample outputs, losses, and statistics are pushed to tensorboard at the end of each "epoch". Currenly these outputs are taken directly from the training data, to avoid opening up a separate data file containing validation data
* Model will automatically restart from a checkpoint *if and only if* there exists a saved checkpoint file in the experiment directory for the run
* The random rotations in the data pipeline degrade the DDP scaling when running on a full GPU node (8 GPUs), currently trying to fix this behavior. Without these, the scaling is close to ideal, even up to 8 nodes/64 GPUs.
