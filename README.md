# [ICLR 2024] Label-Agnostic Forgetting: A Supervision-Free Unlearning in Deep Models

## Abstract
This is a PyTorch implementation of [Label-Agnostic Forgetting] https://openreview.net/forum?id=SIZWiya7FE

Machine unlearning aims to remove information derived from forgotten data while preserving that of the remaining dataset in a well-trained model. With the increasing emphasis on data privacy, several approaches to machine unlearning have emerged. However, these methods typically rely on complete supervision throughout the unlearning process. Unfortunately, obtaining such supervision, whether for the forgetting or remaining data, can be impractical due to the substantial cost associated with annotating real-world datasets. This challenge prompts us to propose a supervision-free unlearning approach that operates without the need for labels during the unlearning process. Specifically, we introduce a variational approach to approximate the distribution of representations for the remaining data. Leveraging this approximation, we adapt the original model to eliminate information from the forgotten data at the representation level. To further address the issue of lacking supervision information, which hinders alignment with ground truth, we introduce a contrastive loss to facilitate the matching of representations between the remaining data and those of the original model, thus preserving predictive performance. Experimental results across various unlearning tasks demonstrate the effectiveness of our proposed method, Label-Agnostic Forgetting (LAF) without using any labels, which achieves comparable performance to state-of-the-art methods that rely on full supervision information. Furthermore, our approach excels in semi-supervised scenarios, leveraging limited supervision information to outperform fully supervised baselines. This work not only showcases the viability of supervision-free unlearning in deep models but also opens up a new possibility for future research in unlearning at the representation level.

<img width="800" alt="framework" src="https://github.com/ShaofeiShen768/LAF/assets/69141552/cad814c6-62ad-44a3-81a0-646ff55a5f6f">


## Environment Setting

Install the camu environment and install all necessary packages:

    conda env create -f laf.yaml

## Dataset Download:  

The data folder includes the experiment data on Mnist, Fashion-Mnist, Cifar10 and SVHN datasets, you can run the code to download them via Torchvision.

## Results Reproduce:  

The hyperparameters for each scenario are recorded in the [hyperparameter.md](hyperparameter.md) file for different models and datasets.

You can run the [LAF.ipynb](LAF.ipynb) file to reproduce the reported results.

The records of training loss have been stored in [Training](Training) folder.

## License

This project is under the MIT license. See [LICENSE](License) for details.

## Citation

@inproceedings{  
shen2024labelagnostic,  
title={Label-Agnostic Forgetting: A Supervision-Free Unlearning in Deep Models},  
author={Shaofei Shen and Chenhao Zhang and Yawen Zhao and Alina Bialkowski and Weitong Tony Chen and Miao Xu},  
booktitle={The Twelfth International Conference on Learning Representations},  
year={2024},  
}
