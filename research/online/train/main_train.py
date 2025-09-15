
import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import multiprocessing as mp

from transformers import AutoModelForCausalLM
import argparse
import json
import logging
import multiprocessing as mp
import queue
import random
import time
from math import ceil
from pathlib import Path
from typing import Optional, Union
from safetensors.torch import save_file
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors
from torch import nn, optim
from torch.utils.data import DataLoader, IterableDataset as TorchIterableDataset

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

from model.configs import EConfig
from model.llama_eagle3_full_grad import Model
import argparse
import json
import os
import random
import warnings
from typing import Any
from utils import AddUniformNoise, AddGaussianNoise, DataCollatorWithPadding, top_accuracy, compute_loss
import numpy as np
import safetensors
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from model.configs import EConfig
from model.llama_eagle3_full_grad import Model
from safetensors import safe_open
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




parser = argparse.ArgumentParser()
parser.add_argument("--basepath", type=str, default=None)
parser.add_argument("--configpath", type=str, default=None)
parser.add_argument("--lr", type=float, default=8e-5)
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument("--tmpdir", type=str, default=None)
parser.add_argument("--cpdir", type=str, default=None)
parser.add_argument("--epoch", type=int, default=4)
parser.add_argument("--topk", type=int, default=10)
parser.add_argument("--topk_w", type=float, default=1.0)
parser.add_argument("--forward_num_total", type=int, default=3)
parser.add_argument("--num_data_workers", type=int, default=4)
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--data_num", type=int, default=8000)
parser.add_argument("--debug", action="store_true")
parser.add_argument('--data_path' ,type=str, default='ShareGPT_V4.3_unfiltered_cleaned_split.json')
args = parser.parse_args()




def build_ds(
    tokenizer,
    split="train",
    worker_id=0, 
):
    total_workers=args.num_data_workers
    ds = load_dataset("json", data_files=args.data_path)

    chunk_size=int(args.data_num/total_workers)
    start=chunk_size*worker_id
    end=chunk_size*(worker_id+1)
    print("start:{}, end:{}, worker_id:{}".format(start, end, worker_id))
    ds = ds[split]
    print(args.data_num)
    ds.select(range(0, args.data_num))
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(start, end))

    original_columns1 = ds1.column_names

    def preprocess(examples):
        new_examples = {"conversation": [], "input_ids": [], "loss_mask": []}
        for j in range(len(examples["id"])):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024"
                    ),
                },
            ]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples["conversations"][j]
            if roles[source[0]["from"]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            for _, sentence in enumerate(source):
                role = roles[sentence["from"]]

                if sentence["from"] == "gpt":
                    sentence["value"] = " " + sentence["value"]
                messages.append({"role": role, "content": sentence["value"]})
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=4096,
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # Ignore the user instructions
                if i == 0:
                    loss_mask[cur_len : cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3 : cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if i != 0:
                    cur_len += 3

            loss_mask[cur_len:] = 0

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess,
        batched=True,
        remove_columns=original_columns1,
        load_from_cache_file=False,
    )

    ds1.set_format(type="torch")
    return ds1





@torch.no_grad()
def ge(data, bigmodel):
    input_ids = data["input_ids"]
    num_layers = len(bigmodel.model.layers)
    with torch.no_grad():
        outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    feature_fusion = [
        outs_big.hidden_states[3],
        outs_big.hidden_states[num_layers // 2 + 1],
        outs_big.hidden_states[-3],
    ]
    target = outs_big.hidden_states[-1]
    hidden_state_big = torch.cat(feature_fusion, dim=-1)

    return {
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0],
        "target": target.cpu()[0],
    }



def online_data_generator(data_queue: mp.Queue,
    shutdown_event: mp.Event,
    verifier: str,
    data_path: str,
    batch_size: int = 32,
    gpu_ids: Optional[list[int]] = None,
    worker_id: int = 0,  # Add worker ID parameter
    num_workers: int = 1, # Add number of workers parameter
):
    print("online data gen")

    bigname = args.basepath
    bigtokenizer = AutoTokenizer.from_pretrained(bigname, use_fast=False)
    ds = build_ds(bigtokenizer, worker_id=worker_id)
    bigmodel = AutoModelForCausalLM.from_pretrained(
        bigname, device_map="auto", torch_dtype=torch.float16
    )
    bigmodel.eval()

    print("loaded model and ready to generate")
    while True:
        for item, data in enumerate(ds):
            outdata = ge(data, bigmodel)
            data_queue.put(outdata)


data_num=args.data_num
print(f"training on {data_num} examples total")
train_frac = 1.0
total_steps = int(
    data_num
    * train_frac
    * (args.epoch + 1)
    / (args.bs * args.gradient_accumulation_steps)
)
warm_steps = total_steps // 100



train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": args.epoch,
    # Depending on your data and model size, the larger the model,
    # the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": warm_steps,
    "total_steps": total_steps,
    "p_w": 0.0,
    "v_w": 0.0,
    "kldiv_w": 1.0,
    "topk_w": args.topk_w,
    "head_w": 0.1,
    "num_workers": 8,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    # During training, truncating the training
    # sequences means that the larger the setting,
    # the more training data is used, and the better the effect,
    # but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 1.0,
    "save_freq": 5,
}
torch.backends.cuda.matmul.allow_tf32 = True

warnings.filterwarnings(
    "ignore", message="You are using `torch.load` with `weights_only=False`*."
)

class CustomDataset(Dataset):  #FLAG    FOR      Later
    def __init__(self, data_queue, transform=None):
        self.data_queue = data_queue
        self.transform = transform

    def __len__(self):
        return args.data_num

    def __getitem__(self, index):
        # data = torch.load(self.data[index])

        data=self.data_queue.get()
        new_data = {}
        hidden_state = data["hidden_state"][: train_config["max_len"]][None, :]

        input_ids = data["input_ids"][: train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][: train_config["max_len"]][None, :]
        target = data["target"][: train_config["max_len"]][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = target[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target

        if self.transform:
            new_data = self.transform(new_data)

        return new_data




def train(data_queue, tr_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, tr_ids))
    data_num = args.data_num
    print(f"training on {args.data_num} examples total")
    train_frac = 1.0
    total_steps = int(
        data_num
        * train_frac
        * (args.epoch + 1)
        / (args.bs * args.gradient_accumulation_steps)
    )
    warm_steps = total_steps // 100

    accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
    )


    baseconfig = AutoConfig.from_pretrained(args.basepath)
    try:
        head = torch.nn.Linear(
            baseconfig.hidden_size, baseconfig.vocab_size, bias=False
        )
    except:
        head = torch.nn.Linear(
            baseconfig.text_config.hidden_size,
            baseconfig.text_config.vocab_size,
            bias=False,
        )

    try:
        with open(os.path.join(args.basepath, "model.safetensors.index.json")) as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        with safe_open(
            os.path.join(args.basepath, head_path), framework="pt", device="cpu"
        ) as f:
            tensor_slice = f.get_slice("lm_head.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except:
        with open(os.path.join(args.basepath, "pytorch_model.bin.index.json")) as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(args.basepath, head_path))
        tensor = weights["lm_head.weight"].float()

    head.weight.data = tensor
    head.eval()

    for param in head.parameters():
        param.requires_grad = False


    if train_config["data_noise"]:
        if train_config["noise"] == "uniform":
            aug = AddUniformNoise(std=train_config["std"])
        else:
            aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
    else:
        aug = None



    traindataset = CustomDataset(data_queue, transform=aug)

    train_loader = DataLoader(
        traindataset,
        batch_size=train_config["bs"],
        shuffle=True,
        collate_fn=DataCollatorWithPadding(),
        num_workers=train_config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    tqdm(train_loader)
    if accelerator.is_main_process and (not os.path.exists(args.cpdir)):
        os.makedirs(args.cpdir)

    config = EConfig.from_pretrained(train_config["config_path"])
    model = Model(config, load_emb=True, path=args.basepath)
    model=model.to(torch.bfloat16)
    if args.ckpt_path is not None:
        ea_model_path = args.ckpt_path
        load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
        if os.path.exists(load_model_path):
            ea_layer_state_dict = torch.load(load_model_path, map_location="cuda")
        else:
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            ea_layer_state_dict = safetensors.torch.load_file(load_model_path)
        model.load_state_dict(ea_layer_state_dict, strict=True)
        print(f"load model from {load_model_path}")
    kldiv = nn.KLDivLoss(reduction="none")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        betas=(train_config["b1"], train_config["b2"]),
    )

    num_epochs = train_config["num_epochs"]
    num_warmup_steps = train_config["num_warmup_steps"]
    total_steps = train_config["total_steps"]
    is_warmup = train_config["is_warmup"]


    if is_warmup:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        )

        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )

    map_tok = np.load("t2d.npy")
    map_tok = torch.from_numpy(map_tok).bool()
    head=head.to(torch.bfloat16)  #MOD - changed to bfloat 16 
    head = head.to(accelerator.device)

    ##MAIN TRAINING LOOP
    for epoch in range(num_epochs + 1):
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.train()
        forward_num_total = args.forward_num_total
        for _batch_idx, data in enumerate(tqdm(train_loader, dynamic_ncols=True)):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                hidden_states, input_ids, attention_mask, target, loss_mask = (
                    data["hidden_states"],
                    data["input_ids"],
                    data["attention_mask"],
                    data["target"],
                    data["loss_mask"][..., None],
                )
                loss = 0
                with torch.no_grad():
                    target_head = head(target.to(torch.bfloat16))

                    target_head = target_head[:, :, map_tok]

                    target_p = nn.Softmax(dim=2)(target_head)
                    target_p = target_p.detach()
                hidden_states_history = []

                hidden_states = model.fc(hidden_states.to(torch.bfloat16))
                weight_sum = 0
                for forward_idx in range(forward_num_total):
                    predict = model(
                        hidden_states, input_ids, attention_mask, hidden_states_history
                    )
                    pred = model.lm_head_layernorm(predict.to(torch.bfloat16))
                    pred = model.lm_head(pred)

                    out_head, kldiv_loss = compute_loss(
                        target_p, pred, loss_mask, kldiv
                    )
                    total_loss = train_config["kldiv_w"] * kldiv_loss
                    weight = forward_idx + 1
                    loss += total_loss
                    weight_sum += weight
                    hidden_states_history.append(hidden_states)
                    hidden_states = torch.concat(
                        [hidden_states[:, :1, :], predict[:, :-1, :]], dim=1
                    )
                accelerator.backward(loss)
                torch.cuda.empty_cache()
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                if is_warmup:
                    scheduler.step()

            with torch.no_grad():
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[
                    loss_mask.view(-1) == 1
                ]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            if accelerator.is_main_process and ct != 0:
                logdict = {
                    "train/lr": optimizer.optimizer.param_groups[0]["lr"],
                    "train/loss": loss.item(),
                    "train/acc": cc / ct,
                }
                for item, _i in enumerate(top_3acc):
                    logdict[f"train/top_{item + 1}_acc"] = topkacc[item].item() / ct

            epoch_loss += loss.item()
            num_batches += 1

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))

        correct, total = correct.sum().item(), total.sum().item()

        epoch_loss /= num_batches
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            print(f"Train Accuracy: {100 * correct / total:.2f}%")

            unwrapped_model = accelerator.unwrap_model(model)
            # torch.save(
            #     unwrapped_model.state_dict(), f"{args.cpdir}/model{epoch}.safetensors"
            # )      
            model_state=unwrapped_model.state_dict()
            save_file(model_state, f"saveout/model{epoch}.safetensors")  

def main(
    verifier: str,
    data: str,
    verifier_batch_size: int,
    data_cache_limit: int = 1000,
    verifier_gpus: Union[float, int, list[int]] = [0, 1, ],
    train_gpus: Union[float, int, list[int]] = [2,3,4, 5, 6, 7],
):

    if mp.current_process().name == "MainProcess":
        print(f"training on {args.data_num} examples total")
        steps_per_epoch = ceil(args.data_num / (args.bs * max(1, args.gradient_accumulation_steps)))

    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")

    data_queue = ctx.Queue(maxsize=int(data_cache_limit))
    shutdown_event = ctx.Event()


    num_workers = args.num_data_workers

    gen_procs = []

    gpus_per_worker = max(1, len(verifier_gpus) // num_workers)


    for i in range(num_workers):
        # Assign GPUs to each worker
        worker_gpus = verifier_gpus[i*gpus_per_worker:(i+1)*gpus_per_worker]
        proc = ctx.Process(
            target=online_data_generator,
            args=(data_queue, shutdown_event, verifier, data, verifier_batch_size,
                  worker_gpus, i, num_workers),daemon=False,
                 )

        proc.start()
        gen_procs.append(proc)
    


    train(data_queue, train_gpus)



    shutdown_event.set()
    for proc in gen_procs:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()
            proc.join()



    return
import time 
if __name__ == "__main__":
    main(
        verifier=args.basepath,
        data="ShareGPT_V4.3_unfiltered_cleaned_split.json",
        verifier_batch_size=1,
    )
    time.sleep(1)