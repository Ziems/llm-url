## LLM-URL: Large Language Models are Built-in Autoregressive Search Engines

### Introduction & Setup

This repository contains the code for the paper [Large Language Models are Built-in Autoregressive Search Engines](https://arxiv.org/abs/2305.09612)(Accepted to ACL 2023).

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
@inproceedings{ziems-2023-large,
    title = "Large Language Models are Built-in Autoregressive Search Engines",
    author = "Ziems, Noah  and
      Yu, Wenhao  and
      Zhang, Zhihan  and
      Jiang, Meng",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    year = "2023"
}
```
Please feel free to cite our paper if you find this repository helpful in your research.
