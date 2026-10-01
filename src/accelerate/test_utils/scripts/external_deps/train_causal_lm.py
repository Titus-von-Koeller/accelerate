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

"""Train a tiny causal LM for distributed training comparisons."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.test_utils.causal_lm import load_model_and_data, observe_backend, register_mixed_precision_check
from accelerate.utils import gather_object, set_seed


def train_reference(model, optimizer, dataloader, mixed_precision_dtype, device):
    """Single-device PyTorch; bypass Accelerate's preparation, backward and optimizer wrappers."""
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision_dtype == torch.float16)
    losses = []

    for batch in dataloader:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=mixed_precision_dtype, enabled=mixed_precision_dtype is not None):
            loss = model(input_ids=batch, labels=batch).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        losses.append(loss.item())

    return losses


def train_with_accelerate(model, optimizer, dataloader, accelerator):
    losses, window_losses = [], []

    for batch in dataloader:
        with accelerator.accumulate(model):
            loss = model(input_ids=batch, labels=batch).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            window_losses.append(loss.detach())
            if accelerator.sync_gradients:
                # Equal shifted-target counts make this a global effective-batch mean.
                window_loss = torch.stack(window_losses).mean()
                losses.append(accelerator.reduce(window_loss, reduction="mean").item())
                window_losses.clear()

    return losses


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--mixed-precision", choices=("no", "bf16", "fp16"), default="no")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    accelerator = None
    if not args.reference:
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    device = torch.device("cuda:0") if args.reference else accelerator.device

    set_seed(1337)
    # Explicitly use full FP32 matmul precision for this comparison.
    torch.set_float32_matmul_precision("highest")

    model, input_ids = load_model_and_data()

    mixed_precision_dtype = {
        "no": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.mixed_precision]
    if mixed_precision_dtype is not None:
        register_mixed_precision_check(model, mixed_precision_dtype)

    world_size = 1 if args.reference else accelerator.num_processes
    if args.batch_size * world_size * args.gradient_accumulation_steps != 8:
        raise ValueError("This fixture requires eight blocks per complete update.")
    if args.reference and args.gradient_accumulation_steps != 1:
        raise ValueError("The PyTorch reference does not implement accumulation.")
    dataloader = DataLoader(input_ids, batch_size=args.batch_size, shuffle=False)

    if args.reference:
        model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    if args.reference:
        losses = train_reference(model, optimizer, dataloader, mixed_precision_dtype, device)
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        losses = train_with_accelerate(model, optimizer, dataloader, accelerator)

    # Revisit the first global batch to observe learning, including the final update.
    first_global_batch = input_ids[:8].to(device)
    # Accelerate's prepared model handles autocast inside forward; only the reference needs it here.
    context = (
        torch.autocast("cuda", dtype=mixed_precision_dtype)
        if args.reference and mixed_precision_dtype is not None
        else nullcontext()
    )
    with torch.no_grad(), context:
        final_loss = model(input_ids=first_global_batch, labels=first_global_batch).loss.item()

    ranks = [] if args.reference else gather_object([observe_backend(accelerator, model)])
    if args.reference or accelerator.is_main_process:
        results = {
            "ranks": ranks,
            "losses": losses,
            "final_loss": final_loss,
            "world_size": 1 if args.reference else accelerator.num_processes,
        }
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    if accelerator:
        accelerator.end_training()


if __name__ == "__main__":
    main()
