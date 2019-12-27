# AdaND
Codebase for the EMNLP19 paper "[Adaptive Parameterization for Neural Dialogue Generation](https://www.aclweb.org/anthology/D19-1188/)".

This codebase is built upon the [ParlAI](https://parl.ai/) project. Check `parlai/agents/AdaND` for our model implementations.

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
