# LLM-fine-tuning
Comparing different types of fine tuning on LLMs

## Install

Install pytorch and other requirements.

- Change directory to `LLM-fine-tuning/`;

- Create the environment in conda with python 3.11:
```bash
conda create -n LLMFineTuning python=3.11 -y
conda activate LLMFineTuning
```

```bash
pip install -r requirements.txt
```

## Run

To train the model:

```
python scripts/gemma-2b-ft.py
```

To evaluate the same model
```
python scripts/eval.py
```

10 steps: 
Train score: 49.46
Eval score: 50.99
20 steps:
