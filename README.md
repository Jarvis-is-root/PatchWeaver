# PatchWeaver

This repository contains the full implementation of the PatchWeaver model, as well as other baseline models for comparative experiments.

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