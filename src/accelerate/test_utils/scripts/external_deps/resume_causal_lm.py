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

"""Save or resume FP32 training at an optimizer-update boundary."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator
from accelerate.test_utils.causal_lm import load_model_and_data
from accelerate.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--mixed-precision", choices=("no",), default="no")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save-at", type=int)
    parser.add_argument("--resume-at", type=int, default=0)
    args = parser.parse_args()
    if args.save_at is not None and args.resume_at:
        parser.error("Save and resume are separate fresh-process runs.")
    if (args.save_at is not None or args.resume_at) and args.checkpoint is None:
        parser.error("Save and resume require a checkpoint directory.")
    if args.save_at is not None and not 0 < args.save_at < 10:
        parser.error("The save point must be between updates 1 and 9.")
    if not 0 <= args.resume_at < 10:
        parser.error("The resume point must be between updates 0 and 9.")

    # The scheduler advances once per completed global update, independently of world size.
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
    )
    if args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps != 8:
        raise ValueError("This fixture requires eight blocks per complete update.")
    set_seed(1337)
    torch.set_float32_matmul_precision("highest")
    model, input_ids = load_model_and_data()
    dataset = TensorDataset(input_ids, torch.arange(len(input_ids)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Momentum makes loading model weights alone insufficient for correct continuation.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    if args.resume_at:
        accelerator.load_state(args.checkpoint)
        # The saved point counts global updates; the prepared loader yields local microbatches.
        dataloader = accelerator.skip_first_batches(dataloader, args.resume_at * args.gradient_accumulation_steps)

    losses, learning_rates, sample_ids = [], [], []
    window_losses, window_ids = [], []
    for batch, ids in dataloader:
        with accelerator.accumulate(model):
            learning_rate = optimizer.param_groups[0]["lr"]
            loss = model(input_ids=batch, labels=batch).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            window_losses.append(loss.detach())
            window_ids.append(ids)
            if accelerator.sync_gradients:
                window_loss = torch.stack(window_losses).mean()
                losses.append(accelerator.reduce(window_loss, reduction="mean").item())
                learning_rates.append(learning_rate)
                sample_ids.append(sorted(accelerator.gather(torch.cat(window_ids)).cpu().tolist()))
                scheduler.step()
                window_losses.clear()
                window_ids.clear()
        if args.save_at is not None and len(losses) == args.save_at:
            accelerator.save_state(args.checkpoint)
            break

    # Both processes revisit the first global batch, as in the ordinary training comparison.
    first_global_batch = input_ids[:8].to(accelerator.device)
    with torch.no_grad():
        final_loss = model(input_ids=first_global_batch, labels=first_global_batch).loss.detach()
    if accelerator.is_main_process:
        result = {
            "losses": losses,
            "final_loss": final_loss.item(),
            "learning_rates": learning_rates,
            "sample_ids": sample_ids,
            "world_size": accelerator.num_processes,
        }
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    accelerator.end_training()


if __name__ == "__main__":
    main()
