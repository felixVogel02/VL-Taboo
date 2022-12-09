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

## Usage when using CUB:

1. Change the datapaths in load_dataset/cub/load_dataset.py to the paths where your download of the CUB dataset is.
2. Execute the experiment by executing a function in experiments/cub/exp.py:
    ```
    experiment1("mypath/results/", module=clip, model_name="clip", name_add="clip")
    ```
    To improve performance you can presave the encoded images by using the functions in imageEncodings/save_imageEncoding_CUB.py

## Usage when using AWA2:

Attributes are only provided per class not per image. Therefore, with the help of a model image attributes are created.

1.  Go in awa2_image_attributes/save_imageLabels.py. There replace all datapaths with the paths to your local AWA2 download and replace the saving paths.Then execute the functions main, after_care, and after_care2 in this ordering to create the image attributes.
2.   Execute the experiment by executing a function in experiments/awa2/exp.py:
    ```
    experiment1("mypath/results/", module=clip, model_name="clip", name_add="clip")
    ```
    For AWA2 there are two different sentence generation techniques. When using the function "experiment1" the sentences are created with the attribtues as a comma seperated list.
    When using "experiment1_new_sent" instead the attributes are inserted in a more complex and natural way into a sentence.
    To improve performance you can presave the encoded images by using the functions in imageEncodings/save_imageEncoding_AWA2.py



