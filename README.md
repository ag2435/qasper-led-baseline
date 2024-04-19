# QASPER benchmark

Baselines tested in [qasper.ipynb](qasper.ipynb).

Differences of our implementation over the original implementation:
1. We use the dataset provided at https://huggingface.co/datasets/allenai/qasper since it doesn't require manually downloading files.
2. We remove usage of `allennlp` since the Python package cannot be installed anymore.
3. We add baselines to [qasper/models](qasper/models/). Currently, we have
    - QASPER (Longformer Encoder Decoder)
    - GPT-3.5-Turbo
    - TODO: RAG (with R=TF-IDF or Contriever) implemented in LangChain?
4. We replace `allennlp` special tokens with the special tokens of the HF transformer tokenizer:
    - paragraph separator: '</s>' -> tokenizer.sep_token
    - sequence pair start tokens: _tokenizer.sequence_pair_start_tokens -> tokenizer.bos_token

## Setup

```
# create conda env
conda env create -f environment.yml

# install the qasper library
conda develop .
```

Dependencies:
```
pytorch
transformers
datasets
```

Data: [original data](https://allenai.org/data/qasper)

## Experiments (from original repo)

### With evidence selection scaffold

**Albert: Is this implemented?**

The configuration file to use is `training_config/led_base_with_evidence_scaffold.jsonnet`. Remember to set the data paths before training.

```
allennlp train training_config/led_base_with_evidence_scaffold.jsonnet -s <PATH TO SERIALIZATION DIRECTORY> --include-package qasper_baselines
```

At the end of training, you will see results on the development set. `best_validation_answer_f1` and `best_validation_evidence_f1` should give you the `Answer F1` and `Evidence F1` reported in the paper.

If you do not have a GPU, you will need to set `cuda_device` to `-1`.


### Without evidence scaffold

Just set `use_evidence_scaffold` in the `model` section of the configuration to `false`.


### Experiments on shorter contexts

The paper also reports results of training and evaluating models given contexts shorter than the full text of the paper. Use the configuration file `training_config/led_base_smaller_context.jsonnet` for these experiments, and set the `context` field in the `dataset_reader` and `validation_dataset_reader` sections of the configuration to appropriate values. 

### Heuristic evidence baselines

The script `scripts/evidence_retrieval_heuristic_baselines.py` contains these baselines. Just run

```
python scripts/evidence_retrieval_heuristic_baselines.py <PATH TO DEV DATA>
```

You will need to install `sklearn` for this script.

Feel free to open pull requests if find any thing that needs fixing.

### Experiments with LED-large

You can run these by changing the value of `transformer_model` variable to `allenai/led-large-16384`. Note that as stated in the paper, the `answer_f1` value will be very low (less than 20 F1 points).
