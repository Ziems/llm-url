## LLM-URL: Large Language Models are Built-in Autoregressive Search Engines

### Introduction & Setup

This repository contains the code for the paper [Large Language Models are Built-in Autoregressive Search Engines](https://arxiv.org/abs/2305.09612)(Accepted to ACL Findings 2023).

- To get started, clone this repository and install the requirements:

```bash
git clone https://github.com/Ziems/llm-url
cd llm-url
pip install -r requirements.txt
```

- Rename `.env.template` to `.env` then add your openai api key

- Download the NQ and TriviaQA (tqa) datasets from Google drive: (we unified the formats) [\[link\]](https://drive.google.com/drive/folders/1lFFTklW_0HuR53hLpFdLClgfSAhXn_2f?usp=sharing) then put these directories in the `indatasets` directory along with WebQ which is already there.

- Run retrieval (step1) on the dataset
    
```bash
python3 mainfunc.py
  --dataset {dataset}
  --task step1
```

- Run answer generation (step2):
    
```bash
python3 mainfunc.py
  --dataset {dataset}
  --task step2
```

## Citation
```
@misc{ziems2023large,
      title={Large Language Models are Built-in Autoregressive Search Engines}, 
      author={Noah Ziems and Wenhao Yu and Zhihan Zhang and Meng Jiang},
      year={2023},
      eprint={2305.09612},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
Please feel free to cite our paper if you find this repository helpful in your research.