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

## Eval

### Metrics

|Metric         |Measures       |Focus|
|---|---|---|
|ROUGE-1	    |Single Words	|Content coverage (Did it mention the right entities?)|
|ROUGE-2	    |Word Pairs	    |Fluency (Did it use natural phrases?)|
|ROUGE-L	    |Sequence	    |Sentence Logic (Did it construct the sentence correctly?)|
|ROUGE-Lsum	    |Paragraph Seq.	|Document Logic (Did the summary flow correctly?)|
|Bert-Precision	|Paragraph Seq.	|Hallucination (How much of the Model's Output was accurate?)|
|Bert-Recall	|Paragraph Seq.	|Completeness (How much of the Reference did the model capture?)|
|Bert-F1    	|Paragraph Seq.	|Both|

```
lm_eval --model hf \
    --model_args pretrained=path/to/your/gemma-2b,dtype=float16 \
    --tasks hellaswag,gsm8k \
    --device cuda:0 \
    --batch_size auto
```