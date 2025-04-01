# Cantonese Phoneme-Level BERT for Enhanced Prosody of Text-to-Speech with Grapheme Predictions (Forked from PL-BERT)

⚠️ **Note:** This is a fork of [PL-BERT](https://github.com/hon9kon9ize/Cantonese-PL-BERT). This fork introduces breaking changes, outlined below.

## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/hon9kon9ize/Cantonese-PL-BERT.git
cd PL-BERT
```
3. Create a new environment (recommended):
```bash
conda create --name BERT python=3.8
conda activate BERT
python -m ipykernel install --user --name BERT --display-name "BERT"
```
4. Install python requirements: 
```bash
pip install pandas singleton-decorator datasets "transformers<4.33.3" accelerate nltk phonemizer sacremoses pebble
```

## Preprocessing
Please refer to the notebook [preprocess.ipynb](https://github.com/hon9kon9ize/Cantonese-PL-BERT/blob/main/preprocess.ipynb) for more details. The preprocessing is for English Wikipedia dataset only. I will make a new branch for Japanese if I have extra time to demostrate training on other languages. You may also refer to [#6](https://github.com/hon9kon9ize/Cantonese-PL-BERT/issues/6#issuecomment-1797869275) for preprocessing in other languages like Japanese. 

## Trianing
Please run train.py to train the PL-BERT model. You can modify the hyperparameters in the config.yml file.

## Finetuning
WIP