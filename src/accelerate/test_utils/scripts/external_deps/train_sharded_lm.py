# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a tiny causal LM for sharded training and checkpoint continuation tests."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from train_causal_lm import register_mixed_precision_check
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator
from accelerate.utils import set_seed


class CheckpointMetadata(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.revision = 0

    def get_extra_state(self):
        return {"revision": self.revision}

    def set_extra_state(self, state):
        self.revision = state["revision"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-at", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume-at", type=int, default=0)
    parser.add_argument("--optimizer", choices=("sgd", "sgd_plain", "adamw"), default="sgd")
    parser.add_argument("--with-extra-state", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = (
        None
        if args.reference
        else Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            step_scheduler_with_optimizer=False,
        )
    )
    device = torch.device("cuda:0") if args.reference else accelerator.device
    set_seed(1337)
    torch.set_float32_matmul_precision("highest")

    checkpoint = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32, use_cache=False)
    metadata = CheckpointMetadata() if args.with_extra_state else None
    if metadata is not None:
        model.add_module("checkpoint_metadata", metadata)
    model.train()
    dtype = {"no": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    if dtype is not None:
        register_mixed_precision_check(model, dtype)

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:100]")
    tokens = tokenizer("\n\n".join(dataset["text"]), return_attention_mask=False)["input_ids"]
    # Equal target counts and complete windows make averaging microbatch losses valid.
    input_ids = torch.tensor(tokens[: 80 * 32]).reshape(80, 32)
    dataloader = DataLoader(input_ids, batch_size=args.batch_size, shuffle=False)
    if args.reference:
        model.to(device)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9 if args.optimizer == "sgd" else 0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if accelerator is not None:
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
        if args.resume_at:
            accelerator.load_state(args.checkpoint)
            # Checkpoints are at optimizer boundaries; this counts prepared-loader microbatches.
            dataloader = accelerator.skip_first_batches(dataloader, args.resume_at * args.gradient_accumulation_steps)

    losses, learning_rates, window_losses = [], [], []
    for batch in dataloader:
        context = nullcontext() if args.reference else accelerator.accumulate(model)
        with context:
            batch = batch.to(device)
            loss = model(input_ids=batch, labels=batch).loss
            if args.reference:
                loss.backward()
            else:
                accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            window_losses.append(loss.detach())
            if args.reference or accelerator.sync_gradients:
                window_loss = torch.stack(window_losses).mean()
                if accelerator is not None:
                    window_loss = accelerator.reduce(window_loss, reduction="mean")
                losses.append(window_loss.item())
                learning_rates.append(optimizer.param_groups[0]["lr"])
                scheduler.step()
                window_losses.clear()
        if args.save_at and len(losses) == args.save_at:
            if metadata is not None:
                metadata.revision = args.save_at
            accelerator.save_state(args.checkpoint)
            break

    # All ranks participate: sharded forward passes can require collectives.
    model.eval()
    first_batch = input_ids[:8].to(device)
    with torch.no_grad():
        final_loss = model(input_ids=first_batch, labels=first_batch).loss.detach()
    final_losses = [final_loss.item()] if args.reference else accelerator.gather(final_loss[None]).tolist()
    if args.reference or accelerator.is_main_process:
        result = {
            "losses": losses,
            "learning_rates": learning_rates,
            "final_losses": final_losses,
            "world_size": 1 if args.reference else accelerator.num_processes,
            "backend": "reference" if args.reference else accelerator.distributed_type.value,
        }
        if metadata is not None:
            result["extra_state_revision"] = metadata.revision
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if accelerator is not None:
        accelerator.end_training()


if __name__ == "__main__":
    main()
