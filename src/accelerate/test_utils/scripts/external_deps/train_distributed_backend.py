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

"""Train a tiny causal LM for the FSDP and DeepSpeed training scenarios."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator, DistributedType
from accelerate.utils import gather_object, set_seed


NUM_UPDATES = 6
EFFECTIVE_BATCH_SIZE = 8


class TrainingBlocks(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "sample_id": index}


def register_mixed_precision_observation(model, observed_dtypes):
    """Observe one representative operation because matching losses do not prove autocast ran."""

    def observe_output_dtype(module, inputs, output):
        observed_dtypes.add(str(output.dtype))

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(observe_output_dtype)
            return

    raise ValueError("Expected a Linear layer to observe mixed-precision output.")


def load_training_blocks(tokenizer):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:100]")
    token_ids = tokenizer("\n\n".join(dataset["text"]), return_attention_mask=False)["input_ids"]

    block_size = 32
    num_blocks = NUM_UPDATES * EFFECTIVE_BATCH_SIZE
    return torch.tensor(token_ids[: num_blocks * block_size]).reshape(num_blocks, block_size)


def backend_observation(accelerator):
    observation = {
        "distributed_type": accelerator.distributed_type.value,
        "world_size": accelerator.num_processes,
    }
    if accelerator.distributed_type == DistributedType.FSDP:
        observation["fsdp_version"] = accelerator.state.fsdp_plugin.fsdp_version
    elif accelerator.distributed_type == DistributedType.DEEPSPEED:
        observation["zero_stage"] = accelerator.state.deepspeed_plugin.zero_stage
    return observation


def train_reference(model, optimizer, dataloader, mixed_precision_dtype, device, max_updates):
    losses, learning_rates, sample_ids = [], [], []
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision_dtype == torch.float16)

    for update, batch in enumerate(dataloader):
        if update == max_updates:
            break

        ids = batch.pop("sample_id")
        batch = {name: value.to(device) for name, value in batch.items()}
        learning_rates.append(optimizer.param_groups[0]["lr"])
        with torch.autocast("cuda", dtype=mixed_precision_dtype, enabled=mixed_precision_dtype is not None):
            loss = model(**batch, labels=batch["input_ids"]).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        losses.append(loss.item())
        sample_ids.append(sorted(ids.tolist()))

    return losses, learning_rates, sample_ids


def train_distributed(
    model,
    optimizer,
    dataloader,
    accelerator,
    max_updates,
    scheduler,
    start_update,
):
    losses, learning_rates, sample_ids = [], [], []
    window_losses, window_sample_ids = [], []
    batches_to_skip = start_update * accelerator.gradient_accumulation_steps
    active_dataloader = accelerator.skip_first_batches(dataloader, batches_to_skip)

    for batch in active_dataloader:
        with accelerator.accumulate(model):
            ids = batch.pop("sample_id")
            learning_rate = optimizer.param_groups[0]["lr"]
            loss = model(**batch, labels=batch["input_ids"]).loss
            accelerator.backward(loss)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            window_losses.append(loss.detach())
            window_sample_ids.append(ids)
            if accelerator.sync_gradients:
                # Every microbatch has the same number of shifted targets, so the
                # mean is the global effective-batch objective for this fixture.
                local_loss = torch.stack(window_losses).mean()
                losses.append(accelerator.reduce(local_loss, reduction="mean").item())
                global_ids = accelerator.gather(torch.cat(window_sample_ids))
                sample_ids.append(sorted(global_ids.cpu().tolist()))
                learning_rates.append(learning_rate)
                window_losses.clear()
                window_sample_ids.clear()

                if start_update + len(losses) == max_updates:
                    break

    return losses, learning_rates, sample_ids


def state_observation(optimizer, scheduler):
    if scheduler is None:
        return None
    return {
        "learning_rate": optimizer.param_groups[0]["lr"],
        "optimizer_state_entries": len(optimizer.state),
        "scheduler_last_epoch": scheduler.last_epoch,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--mixed-precision", choices=("no", "bf16", "fp16"), default="no")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--with-scheduler", action="store_true")
    parser.add_argument("--max-updates", type=int, default=NUM_UPDATES)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--start-update", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.reference and (args.checkpoint_dir or args.resume_from_checkpoint or args.with_scheduler):
        raise ValueError("The plain-PyTorch reference is only used for uninterrupted parity scenarios.")

    accelerator = None
    if not args.reference:
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    device = torch.device("cuda:0") if args.reference else accelerator.device

    set_seed(1337)
    torch.set_float32_matmul_precision("highest")

    checkpoint = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32, use_cache=False)
    model.train()

    observed_dtypes = set()
    register_mixed_precision_observation(model, observed_dtypes)
    input_ids = load_training_blocks(tokenizer)
    dataloader = DataLoader(TrainingBlocks(input_ids), batch_size=args.batch_size, shuffle=False)

    if args.reference:
        model.to(device)
    optimizer_class = torch.optim.SGD if args.optimizer == "sgd" else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=0.02)

    scheduler = None
    loaded_state = None
    if args.reference:
        mixed_precision_dtype = {"no": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.mixed_precision]
        losses, learning_rates, sample_ids = train_reference(
            model, optimizer, dataloader, mixed_precision_dtype, device, args.max_updates
        )
        ranks = [{"distributed_type": "PYTORCH", "world_size": 1}]
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        if args.with_scheduler:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.25,
                total_iters=NUM_UPDATES,
            )
            accelerator.register_for_checkpointing(scheduler)
        if args.resume_from_checkpoint:
            accelerator.load_state(args.resume_from_checkpoint)
            loaded_state = state_observation(optimizer, scheduler)

        losses, learning_rates, sample_ids = train_distributed(
            model,
            optimizer,
            dataloader,
            accelerator,
            args.max_updates,
            scheduler,
            args.start_update,
        )
        if args.checkpoint_dir:
            accelerator.save_state(args.checkpoint_dir)

        rank = backend_observation(accelerator)
        rank["compute_dtypes"] = sorted(observed_dtypes)
        rank["loaded_state"] = loaded_state
        rank["state"] = state_observation(optimizer, scheduler)
        ranks = gather_object([rank])

    first_global_batch = input_ids[:EFFECTIVE_BATCH_SIZE].to(device)
    mixed_precision_dtype = {"no": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.mixed_precision]
    context = (
        torch.autocast("cuda", dtype=mixed_precision_dtype)
        if args.reference and mixed_precision_dtype is not None
        else nullcontext()
    )
    with torch.no_grad(), context:
        final_loss = model(input_ids=first_global_batch, labels=first_global_batch).loss.detach()
    if accelerator:
        final_loss = accelerator.reduce(final_loss, reduction="mean")

    if args.reference:
        ranks[0]["compute_dtypes"] = sorted(observed_dtypes)
        ranks[0]["loaded_state"] = None
        ranks[0]["state"] = state_observation(optimizer, scheduler)

    if args.reference or accelerator.is_main_process:
        result = {
            "losses": losses,
            "learning_rates": learning_rates,
            "sample_ids": sample_ids,
            "final_loss": final_loss.item(),
            "world_size": 1 if args.reference else accelerator.num_processes,
            "ranks": ranks,
        }
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    if accelerator:
        accelerator.end_training()


if __name__ == "__main__":
    main()
