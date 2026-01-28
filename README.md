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
python scripts/train.py
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

### Results

Using Gemma-2b open-source LLM:

|Method-Steps |ROUGE-1 |ROUGE-2 |ROUGE-L |ROUGE-Lsum |Bert-Precision |Bert-Recall |Bert-F1 | 
|---          |---     |---     |---     |---        |---            |---         |---     |
|Max-Values   |1.0000  |1.0000  |1.0000  |1.0000     |1.0000         |1.0000      |1.0000  |
|Lora-10      |0.4743  |0.2357  |0.4740  |0.4732     |0.9248         |0.9203      |0.9222  |
|Lora-20      |0.4934  |0.2265  |0.4946  |0.4937     |0.9269         |0.9097      |0.9178  |
|Lora-30      |0.5082  |0.3003  |0.5084  |0.5071     |0.9330         |0.9144      |0.9232  |
|Lora-100     |0.5241  |0.3138  |0.5238  |0.5237     |0.9364         |0.9240      |0.9298  |
|Lora-200     |0.5194  |0.3023  |0.5192  |0.5192     |0.9345         |0.9276      |0.9307  |
|Lora-500     |0.5224  |0.3087  |0.5222  |0.5228     |0.9343         |0.9295      |0.9316  |
|Lora-1000    |0.5253  |0.3081  |0.5217  |0.5223     |0.9302         |0.9298      |0.9297  |

This tells us that the model is not overfitting since the quote samples weren't in the train step but the LLM still manages to improve the metrics over the training process. After 100 steps it is useless training since the model does not improve.

Now we will compare different types of quantization and finetuning methods:

- QLora x Lora
- QLora x Dora
- LoftQ x Lora
- LoftQ x Dora

|Model        |Method-Quant |ROUGE-1 |ROUGE-2 |ROUGE-L |ROUGE-Lsum |Bert-Precision |Bert-Recall |Bert-F1 | 
|---          |---          |---     |---     |---     |---        |---            |---         |---     |
|Gemma-2b     |QLora-Dora   |0.0000  |0.0000  |0.0000  |0.0000     |0.0000         |0.0000      |0.0000  |
|TinyLlamav1.1|None-Dora    |0.3917  |0.1200  |0.3909  |0.3917     |0.9235         |0.9107      |0.9167  |
|TinyLlamav1.1|QLora-Dora   |0.3913  |0.1328  |0.3904  |0.3904     |0.9230         |0.9067      |0.9144  |
|TinyLlamav1.1|LoftQ-Dora   |0.4051  |0.1470  |0.4060  |0.4061     |0.9266         |0.9117      |0.9187  |


Must do:
- Run different possible combinations for LLama3.2-1b
- Train bert as reward model (?) to compare the two authors
- Fine tune a completion model as a chat model (possible?)


```
lm_eval --model hf \
    --model_args pretrained=path/to/your/gemma-2b,dtype=float16 \
    --tasks hellaswag,gsm8k \
    --device cuda:0 \
    --batch_size auto
```