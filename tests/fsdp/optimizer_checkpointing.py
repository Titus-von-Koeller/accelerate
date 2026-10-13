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

"""Restore FSDP1 optimizer history, CPU loading destinations, and model metadata."""

import argparse
from pathlib import Path
from unittest.mock import patch

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing import assert_close

from accelerate import Accelerator
from accelerate.test_utils.training import observe_backend


class CheckpointMetadata(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.revision = 0

    def get_extra_state(self):
        return {"revision": self.revision}

    def set_extra_state(self, state):
        self.revision = state["revision"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_args = parser.add_mutually_exclusive_group(required=True)
    checkpoint_args.add_argument("--save-checkpoint", type=Path)
    checkpoint_args.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--optimizer", choices=("sgd", "momentum", "adamw"), default="sgd")
    parser.add_argument("--initial-state", choices=("fresh", "initialized", "partial"), default="fresh")
    args = parser.parse_args()

    accelerator = Accelerator()
    model = torch.nn.Linear(4, 4)
    metadata = CheckpointMetadata()
    model.add_module("checkpoint_metadata", metadata)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    else:
        # Saved history must survive even when a fresh optimizer would not create momentum.
        momentum = 0.9 if args.optimizer == "momentum" and args.save_checkpoint else 0
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=momentum)
    model, optimizer = accelerator.prepare(model, optimizer)
    assert observe_backend(accelerator, model) == {"backend": "FSDP", "world_size": 2, "fsdp_version": 1}

    if args.save_checkpoint:
        loss = model(torch.ones(2, 4, device=accelerator.device)).sum()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if args.optimizer == "sgd":
            assert optimizer.state_dict()["state"] == {}
        else:
            assert optimizer.state_dict()["state"]
        metadata.revision = 7
        accelerator.save_state(args.save_checkpoint)
        torch.save(optimizer.state_dict(), args.save_checkpoint / f"expected_optimizer_{accelerator.process_index}.pt")
    else:
        # These options and any existing buffers must be replaced by the checkpoint.
        for group in optimizer.param_groups:
            group["lr"] = 0.7
            if args.optimizer == "adamw":
                group["betas"] = (0.1, 0.2)
            elif args.initial_state != "fresh":
                group["momentum"] = 0.5
        if args.initial_state != "fresh":
            accelerator.backward(model(torch.zeros(2, 4, device=accelerator.device)).sum())
            optimizer.step()
            optimizer.zero_grad()
            assert optimizer.state_dict()["state"]
            if args.initial_state == "partial":
                # A rank-local emptiness check must not choose different collective paths.
                if accelerator.process_index == 0:
                    optimizer.state.clear()
                else:
                    del optimizer.state[next(iter(optimizer.state))]
                    assert optimizer.state

        convert_optimizer_state = FSDP.optim_state_dict_to_load

        def check_cpu_destinations(model, optim, optim_state_dict):
            # CPU offload must govern DCP destinations even though communication uses NCCL.
            for parameter_state in optim_state_dict["state"].values():
                for value in parameter_state.values():
                    if hasattr(value, "local_shards"):
                        assert all(shard.tensor.device.type == "cpu" for shard in value.local_shards())
                    elif isinstance(value, torch.Tensor):
                        assert value.device.type == "cpu"
            return convert_optimizer_state(model, optim, optim_state_dict)

        with patch.object(FSDP, "optim_state_dict_to_load", side_effect=check_cpu_destinations):
            accelerator.load_state(args.resume_from_checkpoint)
        assert metadata.revision == 7
        expected = torch.load(
            args.resume_from_checkpoint / f"expected_optimizer_{accelerator.process_index}.pt",
            map_location="cpu",
            weights_only=True,
        )
        # Exact history and options distinguish restoration from a fresh optimizer warm-up.
        assert_close(optimizer.state_dict(), expected, rtol=0, atol=0, check_device=False)

    accelerator.end_training()


if __name__ == "__main__":
    main()
