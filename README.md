# AdaND
Codebase for the EMNLP19 paper "[Adaptive Parameterization for Neural Dialogue Generation](https://www.aclweb.org/anthology/D19-1188/)".

This codebase is built upon the [ParlAI](https://parl.ai/) project. Check `parlai/agents/AdaND` for our model implementations.

## Citation
```Tex
@inproceedings{hengyi_emnlp19,
  author    = {Hengyi Cai and Hongshen Chen and Cheng Zhang and Yonghao Song and Xiaofang Zhao and Dawei Yin},
  title     = {Adaptive Parameterization for Neural Dialogue Generation},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural
               Language Processing and the 9th International Joint Conference on
               Natural Language Processing, {EMNLP-IJCNLP} 2019, Hong Kong, China,
               November 3-7, 2019},
  pages     = {1793--1802},
  publisher = {Association for Computational Linguistics},
  year      = {2019},
  url       = {https://doi.org/10.18653/v1/D19-1188},
  doi       = {10.18653/v1/D19-1188},
}
```

## Requirements
- Python3
- Pytorch 1.2 or newer

Dependencies of the core modules are listed in requirement.txt.

## Installing
```
git clone git@github.com:hengyicai/AdaND.git ~/AdaND
cd ~/AdaND; python setup.py develop
echo "export PARLAI_HOME=~/AdaND" >> ~/.bashrc; source ~/.bashrc
```

## Running

```
cd ~/AdaND/data
tar -xzvf adand_data.tar.gz
cd ~/AdaND
python projects/AdaND/train_AdaND.py
```
