# Extremely Low-light Image Enhancement with Scene Text Restoration

##### This repository is the official implementation of our work, which is accepted to ICPR 2022 as an [oral presentation](https://arxiv.org/abs/2204.00630).

## Pre-trained Models
> Please download the data here:
> [SID-Sony files](xxx)
> and then decompress it to the "/data" folder.

## Requirements
To create the conda environment, run:
```setup
conda env create -n elie_str --file elie_str.yml
```

## Training
To train the SID-Sony model, run:
```train
python train_Sony_SID.py
```

## Evaluation
To evaluate the trained model, run:
```eval
python test_Sony_SID.py
```

## Citation
If you use this in your research, please cite our work by using the following BibTeX entry:
```latex
 @article{hsu2022extremely,
  title={Extremely Low-light Image Enhancement with Scene Text Restoration},
  author={Hsu, Pohao and Lin, Che-Tsung and Ng, Chun Chet and Kew, Jie-Long and Tan, Mei Yih and Lai, Shang-Hong and Chan, Chee Seng and Zach, Christopher},
  journal={arXiv preprint arXiv:2204.00630},
  year={2022}
}
```
