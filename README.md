## Pose-to-Motion: Cross-Domain Motion Retargeting with Pose Prior - submission 
PyTorch implementation for Pose-to-Motion: Cross-Domain Motion Retargeting with Pose Prior.
<img src='teaser.png'/>

## Set up environment
To setup a conda environment use these commands
```
conda env create -f environment.yml
conda activate pose2motion
```

## Download data
Download the data from [here](https:/) and extract it to the root directory of the project.

## Training
To train the model for mixamo dataset, run the following command
```
python run_all_wgan_mixamo.py
```

## User Study
check out "user_study.md" for more details.