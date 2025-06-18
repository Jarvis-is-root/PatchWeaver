# PatchWeaver

This repository contains the full implementation of the PatchWeaver model, as well as other baseline models for comparative experiments.
---
Unlike the remarkable success of Transformer architectures in natural language processing, recent developments in time series forecasting have shown that Transformers do not demonstrate clear advantages over linear models or RNN-based approaches. In time series forecasting tasks, attention mechanisms are primarily employed in two ways: one approach treats time series similarly to NLP by partitioning sequences along the temporal dimension and applying embeddings, using attention-based encoders to learn relationships between different time positions; the other transposes the input to leverage encoders for learning inter-variate relationships. We argue that the former fails to fully exploit the capability of attention mechanisms to capture position-agnostic correlations, while the latter lacks the exploration of temporal information. Existing attempts to combine both approaches often suffer from computational inefficiency. Thus, we propose ***PatchWeaver***, a method that effectively harnesses the advantages of attention mechanisms in time series tasks to address these limitations. Our approach ***"weaves"*** patches in time series along both variate and temporal dimensions through attention mechanisms and gated units: we partition sequences into patches of typical local lengths at the temporal scale, explicitly modeling local inter-variate relationships following the fundamental principles of attention mechanisms; we employ simplified gating mechanisms to recursively model inter-patch connections for better temporal relationship capture, and finally obtain results across patches in one step through a Flatten Attention Encoder. ***PatchWeaver*** achieves better results than the existing state-of-the-art Transformer-based models on six mainstream time series forecasting datasets, while remaining competitive compared to other architectural approaches.
![PatchWeaver model structure](./resources/main.pdf)

## Setup

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Jarvis-is-root/PatchWeaver.git
    cd PatchWeaver
    ```

2.  **Create and Activate Conda Environment**

    ```bash
    conda create -n patchweaver python=3.10
    conda activate patchweaver
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## Datasets

The datasets used in this project are located in the `dataset/` directory. This includes:
- ETT (ETTh1, ETTh2, ETTm1, ETTm2)
- Electricity
- Weather

## Running Experiments

To run the experiments, you can use the provided scripts. For example, to run the PatchWeaver model:

```bash
bash scripts/PatchWeaver.sh
```

You can also find run scripts for other models in the `scripts` folder.

## Acknowledgement

*   This library is constructed based on [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
*   We also used code from [SparseTSF](https://github.com/lss-1138/SparseTSF) and [CycleNet](https://github.com/ACAT-SCUT/CycleNet) in our experiments.