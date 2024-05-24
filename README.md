<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> (ICML 2024) Revitalizing Multivariate Time Series Forecasting: Learnable Decomposition with Inter-Series Dependencies and Intra-Series Variations Modeling. </b></h2>
</div>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@inproceedings{
anonymous2024revitalizing,
title={Revitalizing Multivariate Time Series Forecasting: Learnable Decomposition with Inter-Series Dependencies and Intra-Series Variations Modeling},
author={Anonymous},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=87CYNyCGOo}
}
```

## Get Started

1. Install requirements. ```pip install -r requirements.txt```
2. Download data. You can download the all datasets from [datasets](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can replicate the benchmark results of **Leddam** from the paper on the eight datasets by:
   
```python run_script.py```

4. Alternatively, you can use bash commands to individually run scripts in the 'scripts' folder from the command line to obtain results for individual datasets, take Traffic for example, you can use the below command line to obtain the result of **input-96-predict-96**:
   
```bash scripts/Traffic/traffic_96.sh ```

You can find the training history and result under 'logs/' folder
