# Cantonese Phoneme-Level BERT for Enhanced Prosody of Text-to-Speech with Grapheme Predictions (Forked from PL-BERT)

⚠️ **Note:** This is a fork of [PL-BERT](https://github.com/hon9kon9ize/Cantonese-PL-BERT). This fork introduces breaking changes, outlined below.

### 🧠 Why PL-BERT?

PL-BERT (Phoneme-Level BERT) was originally developed to improve prosody in text-to-speech (TTS) systems by modeling phoneme sequences directly. Traditional TTS pipelines often require heavily annotated, structured corpora, which are expensive and difficult to obtain—especially for low-resource languages. PL-BERT solves this by learning contextualized phoneme representations that can be fine-tuned separately from the main TTS model, effectively **decoupling pretraining from task-specific supervised learning**.

---

### 🗣️ Why Cantonese PL-BERT?

Cantonese is a low-resource language in the TTS and NLP landscape. Annotated Cantonese datasets with phoneme-level labels are scarce, and the written form often diverges from its spoken usage. By pretraining a **separate Cantonese PL-BERT model**, we unlock several benefits:

- 🔓 **Decoupling**: Enables pretraining on large amounts of **raw Cantonese text** without requiring phoneme-level annotations.
- 🧩 **Modular Design**: The pretrained PL-BERT can be plugged into downstream TTS systems to improve prosody and intelligibility, reducing reliance on specialized datasets.
- 🌏 **Language Preservation**: Enhances the accessibility and naturalness of Cantonese voice applications, which is crucial for a language with strong spoken culture but limited digital support.

This approach can serve as a **template for other low-resource languages**, where annotated corpora are limited but unstructured text and speech are still available.

---

### Pre-requisites
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

---

### Dataset

The dataset we used is a processed Yue Wikipeda with ToJyutping as phonemizer [hon9kon9ize/yue-wiki-pl-bert](https://huggingface.co/datasets/hon9kon9ize/yue-wiki-pl-bert).

## Preprocessing
Please refer to the notebook [preprocess.ipynb](https://github.com/hon9kon9ize/Cantonese-PL-BERT/blob/main/preprocess.ipynb) for more details. The preprocessing is for English Wikipedia dataset only. I will make a new branch for Japanese if I have extra time to demostrate training on other languages. You may also refer to [#6](https://github.com/hon9kon9ize/Cantonese-PL-BERT/issues/6#issuecomment-1797869275) for preprocessing in other languages like Japanese. 

---

### Trianing
Please run train.py to train the PL-BERT model. You can modify the hyperparameters in the config.yml file.

---

### Finetuning
WIP
