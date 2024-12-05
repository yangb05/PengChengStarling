# Introduction

The PengChengStarling project is a multilingual ASR system development toolkit built upon [the icefall project](https://github.com/k2-fsa/icefall). Compared to the original icefall, it incorporates several task-specific optimizations for ASR. 
Firstly, we replaced the recipe-based approach with a more flexible design, decoupling parameter configurations from functional code. This allows a unified codebase to support ASR tasks across multiple languages. 
Secondly, we integrated language IDs into the RNNT architecture, significantly improving the performance of multilingual ASR systems.

# Installation

Please refer to the [document](https://icefall.readthedocs.io/en/latest/installation/index.html) for installation instructions. If the installation test is successful, PengChengStarling is ready for use.

# Training
## 1. Data Preparation

Before starting the training process, you must first preprocess the raw data into the required input format. Typically, this involves adapting the `make_\*_list` method in `PengChengStarling/zipformer/prepare.py` to your dataset to generate the `data.list` file. 
The subsequent steps are applicable to all datasets. Once completed, the script will produce the corresponding cuts and fbank features for each dataset, which serve as the input data for PengChengStarling. 
The script’s parameters are configured through YAML files located in the `config_data` directory. Once you’ve prepared the YAML file for your dataset, you can run the script using the following command:

```bash
export CUDA_VISIBLE_DEVICES="0"

python zipformer/prepare.py --config-file config_data/<your_dataset_config>.yaml
```

For more details about cut, please refer to the [document](https://lhotse.readthedocs.io/en/latest/cuts.html).

## 2. BPE Training


