import yaml
import torch
from torch import nn
from transformers import (
    BertTokenizer,
    BertConfig,
    BertModel,
    TrainingArguments,
    Trainer,
)
from model import MultiTaskModel
from datasets import load_from_disk
from dataloader import build_dataloader
from utils import length_to_mask

config_path = "Configs/config_yue.yml"  # you can change it to anything else
config = yaml.safe_load(open(config_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define tokenizer
tokenizer = BertTokenizer.from_pretrained(config["dataset_params"]["tokenizer"])

# define model
bert_base_configuration = BertConfig(**config["model_params"])
bert = BertModel(bert_base_configuration)
bert = torch.compile(bert)
model = MultiTaskModel(
    bert,
    num_vocab=len(tokenizer.get_vocab()),
    num_tokens=config["model_params"]["vocab_size"],
    hidden_size=config["model_params"]["hidden_size"],
)


# define dataset
dataset = load_from_disk(config["data_folder"])

batch_size = config["batch_size"]
train_loader = build_dataloader(
    dataset,
    batch_size=batch_size,
    num_workers=0,
    dataset_config=config["dataset_params"],
)

training_args = TrainingArguments(
    output_dir=config["output_dir"],
    run_name="yue-pl-bert",
    num_train_epochs=10,
    auto_find_batch_size=True,
    logging_strategy="steps",
    logging_steps=200,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    weight_decay=0.05,
    save_safetensors=False,
    save_strategy="epoch",
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": 1.0e-7},
    bf16=True,
    remove_unused_columns=False,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader.dataset,
    data_collator=train_loader.collate_fn,
)


trainer.train()
