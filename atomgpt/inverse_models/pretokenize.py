# pretokenize.py
"""
Takes a dataset in the id_prop.csv format and tokenizes it according to parameters
specified in the config.json. These tokenized train and test sets are saved to
disk, and the relevant metadata for pretokenization are saved alongside the train
and test sets.
"""

from typing import Optional
from atomgpt.inverse_models.loader import FastLanguageModel

# from unsloth import FastLanguageModel
from atomgpt.inverse_models.callbacks import (
    PrintGPUUsageCallback,
    ExampleTrainerCallback,
)
from transformers import (
    TrainingArguments,
)
import torch
from atomgpt.inverse_models.utils import (
    gen_atoms,
    text2atoms,
    get_crystal_string_t,
    get_figlet,
)
from trl import SFTTrainer, SFTConfig
from peft import PeftModel
from datasets import load_dataset
from functools import partial
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson, dumpjson
from tqdm import tqdm
from pydantic import BaseModel
import pprint
from jarvis.io.vasp.inputs import Poscar
import csv
import os
from pydantic_settings import BaseSettings
import sys
import json
import argparse
from typing import Literal
import time
#from datasets import save_to_disk
from jarvis.core.composition import Composition
from atomgpt.inverse_models.inverse_models import TrainingPropConfig
from atomgpt.inverse_models.inverse_models import make_alpaca_json
from atomgpt.inverse_models.inverse_models import formatting_prompts_func

parser = argparse.ArgumentParser(
    description="Atomistic Generative Pre-trained Transformer."
)
parser.add_argument(
    "--config_name",
    default="alignn/examples/sample_data/config_example.json",
    help="Name of the config file",
)

class PretokConfig(BaseModel):
    id_prop_path: str
    tokenizer_class: str
    num_train: int | None = None
    num_test: int | None = None
    test_ratio: float | None = None
    separator: str | None = None
    prop: str
    file_format: str
    alpaca_prompt: str | None = None
    model_name: str
    max_seq_length: int
    dtype: str | None = None
    load_in_4bit: bool | None = None

def make_pretok_config(config, dir=None):
    pretok_dict = {
        "id_prop_path": config.id_prop_path,
        "tokenizer_class": config.tokenizer_class,
        "num_train": config.num_train,
        "num_test": config.num_test,
        "test_ratio": config.test_ratio,
        "separator": config.separator,
        "prop": config.prop,
        "file_format": config.file_format,
        "alpaca_prompt": config.alpaca_prompt,
        "model_name": config.model_name,
        "max_seq_length": config.max_seq_length,
        "dtype": str(config.dtype) if config.dtype is not None else None,
        "load_in_4bit": config.load_in_4bit,
    }
    return PretokConfig(**pretok_dict)

def main(config_file=None):
    if config_file is None:
        args = parser.parse_args(sys.argv[1:])
        config_file = args.config_name
    figlet = get_figlet()
    print(figlet)
    t1 = time.time()
    #print("config_file", config_file)
    config = loadjson(config_file)
    config = TrainingPropConfig(**config)
    id_prop_path = config.id_prop_path
    base_dir = os.path.dirname(os.path.abspath(id_prop_path))
    pretok_dir = os.path.join(base_dir, str(config.tokenizer_class))
    os.makedirs(pretok_dir, exist_ok=True)
    pretok_config = make_pretok_config(config, pretok_dir)
    print("pretokenization parameters", pretok_config)
    pprint.pprint(pretok_config.dict())
    f = open(os.path.join(pretok_dir, "pretok_metadata.json"), "w")
    f.write(json.dumps(pretok_config.dict(), indent=4))
    f.close()
    num_train = config.num_train
    num_test = config.num_test
    with open(id_prop_path, "r") as f:
        reader = csv.reader(f)
        dt = [row for row in reader]
    if not num_train:
        num_test = int(len(dt) * config.test_ratio)
        num_train = len(dt) - num_test

    dat = []
    ids = []
    for i in tqdm(dt, total=len(dt)):
        info = {}
        info["id"] = i[0]
        ids.append(i[0])
        tmp = [float(j) for j in i[1:]]
        # print("tmp", tmp)
        if len(tmp) == 1:
            tmp = str(float(tmp[0]))
        else:
            tmp = config.separator.join(map(str, tmp))

        # if ";" in i[1]:
        #    tmp = "\n".join([str(round(float(j), 2)) for j in i[1].split(";")])
        # else:
        #    tmp = str(round(float(i[1]), 3))
        info[config.prop] = (
            tmp  # float(i[1])  # [float(j) for j in i[1:]]  # float(i[1]
        )
        run_path = os.path.dirname(id_prop_path)
        pth = os.path.join(run_path, info["id"])
        if config.file_format == "poscar":
            atoms = Atoms.from_poscar(pth)
        elif config.file_format == "xyz":
            atoms = Atoms.from_xyz(pth)
        elif config.file_format == "cif":
            atoms = Atoms.from_cif(pth)
        elif config.file_format == "pdb":
            # not tested well
            atoms = Atoms.from_pdb(pth)
        info["atoms"] = atoms.to_dict()
        dat.append(info)

    train_ids = ids[0:num_train]
    print("num_train", num_train)
    print("num_test", num_test)
    test_ids = ids[num_train : num_train + num_test]
    # test_ids = ids[num_train:]
    alpaca_prop_train_filename = os.path.join(
        pretok_dir, "alpaca_prop_train.json"
    )
    if not os.path.exists(alpaca_prop_train_filename):
        m_train = make_alpaca_json(
            dataset=dat,
            jids=train_ids,
            config=config,
            # prop=config.property_name,
            # instruction=config.instruction,
            # chem_info=config.chem_info,
            # output_prompt=config.output_prompt,
        )
        dumpjson(data=m_train, filename=alpaca_prop_train_filename)
    else:
        print(alpaca_prop_train_filename, " exists")
        m_train = loadjson(alpaca_prop_train_filename)
    print("Sample:\n", m_train[0])

    alpaca_prop_test_filename = os.path.join(
        pretok_dir, "alpaca_prop_test.json"
    )
    if not os.path.exists(alpaca_prop_test_filename):

        m_test = make_alpaca_json(
            dataset=dat,
            jids=test_ids,
            config=config,
            # prop="prop",
            include_jid=True,
            # instruction=config.instruction,
            # chem_info=config.chem_info,
            # output_prompt=config.output_prompt,
        )
        dumpjson(data=m_test, filename=alpaca_prop_test_filename)
    else:
        print(alpaca_prop_test_filename, "exists")
        m_test = loadjson(alpaca_prop_test_filename)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    del model
    
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        # model.resize_token_embeddings(len(tokenizer))
    train_dataset = load_dataset(
        "json",
        data_files=alpaca_prop_train_filename,
        split="train",
        # "json", data_files="alpaca_prop_train.json", split="train"
    )
    eval_dataset = load_dataset(
        "json",
        data_files=alpaca_prop_test_filename,
        split="train",
        # "json", data_files="alpaca_prop_train.json", split="train"
    )
    formatting_prompts_func_with_prompt = partial(
        formatting_prompts_func, alpaca_prompt=config.alpaca_prompt
    )

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_length,
        )

    train_dataset = train_dataset.map(
        formatting_prompts_func_with_prompt,
        batched=True,
    )
    eval_dataset = eval_dataset.map(
        formatting_prompts_func_with_prompt,
        batched=True,
    )
    # Compute the actual max sequence length in raw text
    lengths = [
        len(tokenizer(example["text"], truncation=False)["input_ids"])
        for example in eval_dataset
    ]
    max_seq_length = max(lengths)
    print(f"🧠 Suggested max_seq_length based on dataset: {max_seq_length}")

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    tokenized_train.set_format(
        type="torch", columns=["input_ids", "attention_mask", "output"]
    )
    tokenized_eval.set_format(
        type="torch", columns=["input_ids", "attention_mask", "output"]
    )
    
    formatted_train_path = os.path.join(pretok_dir, "formatted_train.jsonl")
    formatted_test_path  = os.path.join(pretok_dir, "formatted_test.jsonl")
    train_dataset.to_json(formatted_train_path)
    eval_dataset.to_json(formatted_test_path)
    
    tokenized_train.save_to_disk(os.path.join(pretok_dir, "train"))
    tokenized_eval.save_to_disk(os.path.join(pretok_dir, "test"))
    

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    main(config_file=args.config_name)
