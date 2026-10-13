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

"""Round-trip an FSDP1 checkpoint with empty optimizer state and model metadata."""

import argparse
from pathlib import Path

import torch

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
    args = parser.parse_args()

    accelerator = Accelerator()
    model = torch.nn.Linear(4, 4)
    metadata = CheckpointMetadata()
    model.add_module("checkpoint_metadata", metadata)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    assert observe_backend(accelerator, model) == {"backend": "FSDP", "world_size": 2, "fsdp_version": 1}

    if args.save_checkpoint:
        loss = model(torch.ones(2, 4, device=accelerator.device)).sum()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        # Plain SGD legitimately has no optimizer tensors, even after an update.
        assert optimizer.state_dict()["state"] == {}
        metadata.revision = 7
        accelerator.save_state(args.save_checkpoint)
    else:
        accelerator.load_state(args.resume_from_checkpoint)
        # Model metadata must survive loading without becoming optimizer tensor-layout hints.
        assert metadata.revision == 7
        # DCP omits the empty dictionary; Accelerate must restore it for FSDP.
        assert optimizer.state_dict()["state"] == {}

    accelerator.end_training()


if __name__ == "__main__":
    main()
