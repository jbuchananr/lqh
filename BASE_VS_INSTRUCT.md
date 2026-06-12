# Which model is a better base for fine-tuning?

We want to create an script with the tests folder that runs a small benchmark to compare the performance of different base models for fine-tuning.

- Create an eval (400 samples) and train dataset (20k samples)
- Run an "local eval" of these models on the eval set with a proper system prompt.
- Run an SFT sweep on this training set. Make sure it's a sweep since hyperparameters are very different for the base vs instruct variants.
- Evaluate the best sweep checkpoint on the eval set.
- Run a DPO sweep on top of the best SFT checkpoint for each of the models. Evaluate the best DPO checkpoint on the eval set.
- Create a full result table with the eval results of the local eval and the best sweep checkpoint for each model.

Models under test from huggingface:
LiquidAI/LFM2.5-350M
LiquidAI/LFM2.5-350M-Base
LiquidAI/LFM2.5-1.2B-Instruct
LiquidAI/LFM2.5-1.2B-Base

Note: Do not use the API eval but local one.


## Possible tasks

Ideally, we don't have just one task but a few to check if the trend is consistent or very dependent on the task.
Some ideas:
- a translation task (non-strandard, eg. EN-FR probably not good)
- structured data extraction (eg. some kind of unstructured data to JSON)
- classification (eg. spam detection, error detection, sentiment analysis)
