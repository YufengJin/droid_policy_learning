# DROID Policy Learning and Evaluation

This repository contains code for training and evaluating policies on the [DROID](https://droid-dataset.github.io) dataset. DROID is a large-scale, in-the-wild robot manipulation dataset. This codebase is built as a fork of [`robomimic`](https://robomimic.github.io/), a popular repository for imitation learning algorithm development. For more information about DROID, please see the following links: 

[**[Homepage]**](https://droid-dataset.github.io) &ensp; [**[Documentation]**](https://droid-dataset.github.io/droid) &ensp; [**[Paper]**](https://arxiv.org/abs/2403.12945) &ensp; [**[Dataset Visualizer]**](https://droid-dataset.github.io/dataset.html).

-------
## Updates

- [04/22/25]: We provide improved camera calibrations for 36k episodes of the DROID dataset [on HuggingFace](https://huggingface.co/KarlP/droid) -- check our [updated paper](https://arxiv.org/abs/2403.12945) for how we computed these calibration values!
- [12/21/24]: We provide an updated set of DROID language annotations [on HuggingFace](https://huggingface.co/KarlP/droid) -- 3 natural language annotations for 95% of all successful DROID episodes (75k episodes)

-------
## Installation

### Option A: Docker (Recommended)

Use the provided Docker setup for a reproducible environment with GPU support:

```bash
docker compose -f docker/docker-compose.headless.yaml build
docker compose -f docker/docker-compose.headless.yaml up -d
docker exec -it droid-dev-headless bash
```

See **[docker/README.md](docker/README.md)** for full installation, configuration, and usage instructions (including optional Robocasa support).

### Option B: Conda (Local)

Create a python3 conda environment (tested with Python 3.10) and run the following:

1. Create python 3.10 conda environment: `conda create --name droid_policy_learning_env python=3.10`
2. Activate the conda environment: `conda activate droid_policy_learning_env`
3. Install [octo](https://github.com/octo-models/octo/tree/main), pinned at commit `85b83fc19657ab407a7f56558a5384ae56fe453b` (used for data loading)
4. Run `pip install -e .` in `droid_policy_learning`.

With this you are all set up for training policies on DROID. If you want to evaluate your policies on a real robot DROID setup, 
please install the DROID robot controller in the same conda environment (follow the instructions [here](https://github.com/droid-dataset/droid)).

-------
## Preparing Datasets
We provide all DROID datasets in RLDS format, which makes it easy to co-train with various other robot-learning datasets (such as those in the [Open X-Embodiment](https://robotics-transformer-x.github.io/)).

To download the DROID dataset from the Google cloud bucket, install the [gsutil package](https://cloud.google.com/storage/docs/gsutil_install) and run the following command (Note: the full dataset is 1.7TB in size):
```
gsutil -m cp -r gs://gresearch/robotics/droid <path_to_your_target_dir>
```

We also provide a small (2GB) example dataset with 100 DROID trajectories that uses the same format as the full RLDS dataset and can be used for code prototyping and debugging:
```
gsutil -m cp -r gs://gresearch/robotics/droid_100 <path_to_your_target_dir>
```

For good performance of DROID policies in your target setting, it is helpful to include a small number of demonstrations in your target domain into the training mix ("co-training"). 
Please follow the instructions [here](https://droid-dataset.github.io/droid/example-workflows/data-collection.html) for collecting a small teleoperated dataset in your target domain and instructions [here](https://github.com/kpertsch/droid_dataset_builder) converting it to the RLDS training format.
Make sure that all datasets you want to train on are under the same root directory `DATA_PATH`.

*Note*: We also provide the raw DROID dataset at stereo, full HD resolution. If your training pipeline requires this information, you can download the dataset from `gs://gresearch/robotics/droid_raw`. For a detailed description of the raw data format, please see our [developer documentation](https://droid-dataset.github.io/droid).

-------
## Training

### Step 1: Configure Training Parameters

Edit `robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py` and update the following key parameters:

```python
# Line 13: Path to your RLDS datasets directory
DATA_PATH = "/workspace/dataset/droid_100"    # Update this path

# Line 14: Path where training logs and checkpoints will be saved
EXP_LOG_PATH = "/workspace/droid_policy_learning/logs"  # Update this path

# Lines 15-22: Define your experiments and datasets
EXP_NAMES = OrderedDict([
    ("droid", {
        "datasets": ["droid"],           # Dataset names (should match folder names in DATA_PATH)
        "sample_weights": [1]            # Sample weights for each dataset
    })
])
```

**Important Parameters:**
- `DATA_PATH`: Directory containing all RLDS datasets. Each dataset should be in its own subdirectory (e.g., `DATA_PATH/droid/`).
- `EXP_LOG_PATH`: Directory where training logs, checkpoints, and experiment outputs will be stored.
- `EXP_NAMES`: Defines experiment configurations:
  - Keys are experiment names (as logged in wandb)
  - `datasets`: List of dataset folder names in `DATA_PATH`
  - `sample_weights`: Relative sampling weights for each dataset (e.g., `[1, 2]` means second dataset is sampled twice as often)

**For Docker Users:**
- If using the provided Docker setup, datasets are mounted at `/workspace/dataset/`
- Project code is at `/workspace/droid_policy_learning/`
- Default paths in the config file are already set for Docker environment
- **Troubleshooting**: If you encounter GPU access issues or TensorFlow warnings, see [Docker Troubleshooting Guide](docs/docker_troubleshooting.md)

### Step 2: Generate Training Configuration

Run the config generator script to create training configurations:

```bash
cd /workspace/droid_policy_learning  # If in Docker container
python robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py \
    --wandb_proj_name <YOUR_WANDB_PROJECT_NAME> \
    --env droid \
    --mod <MODALITY>  # e.g., "rgb" for vision-based policies
```

This will:
1. Generate JSON config files for each experiment
2. Create a bash script (default: `~/tmp/tmpp.sh`) with training commands

**Example:**
```bash
python robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py \
    --wandb_proj_name droid_policy_training \
    --env droid \
    --mod rgb
```

### Step 3: Launch Training

**Option A: Use the generated script**
```bash
bash ~/tmp/tmpp.sh
```

**Option B: Run training directly**
The generated script contains commands like:
```bash
python /workspace/droid_policy_learning/robomimic/scripts/train.py \
    --config /path/to/generated/config.json
```

You can also run this command directly or modify it as needed.

### Additional Training Parameters

You can customize training by modifying parameters in `droid_runs_language_conditioned_rlds.py`:

- **Batch size** (line 74): `values=[128]` - Adjust based on GPU memory
- **Image resolution** (line 203): `values=[[128, 128]]` - Input image size
- **Number of epochs** (line 53): `values=[100000]` - Training duration
- **Shuffle buffer size** (line 67): `values=[500000]` - Dataset shuffling (reduce if low RAM)
- **Camera selection** (line 214): Choose which cameras to use
- **Observation modalities** (line 274): Configure low-dim and image observations

See the `robomimic` documentation for more information on config parameters.

### Training Tips

- **Shuffle Buffer Size**: We use a [_shuffle buffer_](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle) to ensure training samples are properly randomized. The default is `500000`, but reduce this if you have limited RAM. For best results, use `shuffle_buffer_size >= 100000` if possible.

- **GPU Requirements**: All policies were trained on a single NVIDIA A100 GPU. You may need to adjust batch size and other parameters for different GPU configurations.

- **Weights and Biases (wandb) Logging**: 
  - Update `WANDB_ENTITY` and `WANDB_API_KEY` in `robomimic/macros.py`
  - Or set environment variables: `export WANDB_ENTITY=your_entity` and `export WANDB_API_KEY=your_key`
  - Training metrics, checkpoints, and visualizations will be logged to wandb

- **Monitoring Training**: 
  - Check wandb dashboard for real-time training metrics
  - Checkpoints are saved to `EXP_LOG_PATH/<experiment_name>/models/`
  - Training logs are saved to `EXP_LOG_PATH/<experiment_name>/logs/`

We also provide a stand-alone example to load data from DROID [here](examples/droid_dataloader.py).

-------
## Code Structure

|                           | File                                                    | Description                                                                   |
|---------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------|
| Hyperparameters           | [droid_runs_language_conditioned_rlds.py](robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py)     | Generates a config based on defined hyperparameters  |
| Training Loop             | [train.py](robomimic/scripts/train.py)                  | Main training script.                                                         |
| Datasets                  | [dataset.py](https://github.com/octo-models/octo/blob/main/octo/data/dataset.py)                      | Functions for creating datasets and computing dataset statistics,             |
| RLDS Data Processing      | [rlds_utils.py](robomimic/utils/rlds_utils.py)    | Processing to convert RLDS dataset into dataset compatible for DROID training                      |
| General Algorithm Class   | [algo.py](robomimic/algo/algo.py)             | Defines a high level template for all algorithms (eg. diffusion policy) to extend           |
| Diffusion Policy          | [diffusion_policy.py](robomimic/algo/diffusion_policy.py)    | Implementation of diffusion policy |
| Observation Processing    | [obs_nets.py](robomimic/models/obs_nets.py)    | General observation pre-processing/encoding |
| Visualization             | [vis_utils.py](robomimic/utils/vis_utils.py) | Utilities for generating trajectory visualizations                      |

-------

## Evaluating Trained Policies
To evaluate policies, make sure that you additionally install [DROID](https://github.com/droid-dataset/droid) in your conda environment and then run:
```python
python scripts/evaluation/evaluate_policy.py
```
from the DROID root directory. Make sure to use the appropriate command line arguments for the model checkpoint path and whether to do goal or language conditioning, and then follow
all resulting prompts in the terminal. To replicate experiments from the paper, use the language conditioning mode.

-------

## Training Policies with HDF5 Format
Natively, robomimic uses HDF5 files to store and load data. While we mainly support RLDS as the data format for training with DROID, [here](README_hdf5.md) are instructions for how to run training with the HDF5 data format.

------------
## Citation

```
@misc{droid_2024,
    title={DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset},
    author  = {Alexander Khazatsky and Karl Pertsch and Suraj Nair and Ashwin Balakrishna and Sudeep Dasari and Siddharth Karamcheti and Soroush Nasiriany and Mohan Kumar Srirama and Lawrence Yunliang Chen and Kirsty Ellis and Peter David Fagan and Joey Hejna and Masha Itkina and Marion Lepert and Yecheng Jason Ma and Patrick Tree Miller and Jimmy Wu and Suneel Belkhale and Shivin Dass and Huy Ha and Arhan Jain and Abraham Lee and Youngwoon Lee and Marius Memmel and Sungjae Park and Ilija Radosavovic and Kaiyuan Wang and Albert Zhan and Kevin Black and Cheng Chi and Kyle Beltran Hatch and Shan Lin and Jingpei Lu and Jean Mercat and Abdul Rehman and Pannag R Sanketi and Archit Sharma and Cody Simpson and Quan Vuong and Homer Rich Walke and Blake Wulfe and Ted Xiao and Jonathan Heewon Yang and Arefeh Yavary and Tony Z. Zhao and Christopher Agia and Rohan Baijal and Mateo Guaman Castro and Daphne Chen and Qiuyu Chen and Trinity Chung and Jaimyn Drake and Ethan Paul Foster and Jensen Gao and Vitor Guizilini and David Antonio Herrera and Minho Heo and Kyle Hsu and Jiaheng Hu and Muhammad Zubair Irshad and Donovon Jackson and Charlotte Le and Yunshuang Li and Kevin Lin and Roy Lin and Zehan Ma and Abhiram Maddukuri and Suvir Mirchandani and Daniel Morton and Tony Nguyen and Abigail O'Neill and Rosario Scalise and Derick Seale and Victor Son and Stephen Tian and Emi Tran and Andrew E. Wang and Yilin Wu and Annie Xie and Jingyun Yang and Patrick Yin and Yunchu Zhang and Osbert Bastani and Glen Berseth and Jeannette Bohg and Ken Goldberg and Abhinav Gupta and Abhishek Gupta and Dinesh Jayaraman and Joseph J Lim and Jitendra Malik and Roberto Martín-Martín and Subramanian Ramamoorthy and Dorsa Sadigh and Shuran Song and Jiajun Wu and Michael C. Yip and Yuke Zhu and Thomas Kollar and Sergey Levine and Chelsea Finn},
    year = {2024},
}
```
