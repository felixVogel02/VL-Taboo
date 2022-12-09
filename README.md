# VL-Taboo
[[PAPER]](https://arxiv.org/abs/2209.06103)

Official implementation for the paper: "VL-Taboo: An Analysis of Attribute-based Zero-shot Capabilities of
Vision-Language Models"

In order to check the capabilities of Vision-Language models handling attributes, this repository makes the code for all five experiments proposed in the VL-Taboo paper publicly available.

The current implementation works for the three models:
* [CLIP](https://github.com/openai/CLIP)
* [OpenCLIP](https://github.com/mlfoundations/open_clip)
* [FLAVA](https://huggingface.co/docs/transformers/model_doc/flava)

and for the two datasets:
* [AWA2](https://paperswithcode.com/dataset/awa2-1)
* [CUB-200-2011](https://paperswithcode.com/dataset/cub-200-2011)

Usage when using CUB:

1. Change the datapaths in load_dataset/cub/load_dataset.py to the paths where your download of the CUB dataset is.
2. Execute the experiment by executing a function in experiments/cub/exp.py:
    ```
    experiment1(module=clip, model_name="clip", name_add="clip")
    ```


