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

"""Launch the sharded training worker in an isolated process."""

import json
import os
import sys

from .testing import execute_subprocess_async, get_torch_dist_unique_port, path_in_accelerate_package


def run_training(
    output,
    *,
    launch_args=(),
    batch_size=4,
    mixed_precision="no",
    gradient_accumulation_steps=1,
    reference=False,
    checkpoint=None,
    save_at=None,
    resume_at=0,
    optimizer="sgd",
):
    command = [sys.executable]
    if not reference:
        command += [
            "-m",
            "accelerate.commands.launch",
            "--num_processes=2",
            "--num_machines=1",
            "--machine_rank=0",
            "--dynamo_backend=no",
            f"--mixed_precision={mixed_precision}",
            f"--main_process_port={get_torch_dist_unique_port()}",
            *launch_args,
        ]
    command += [
        str(path_in_accelerate_package("test_utils", "scripts", "external_deps", "train_sharded_lm.py")),
        "--output",
        str(output),
        "--batch-size",
        str(batch_size),
        "--mixed-precision",
        mixed_precision,
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--optimizer",
        optimizer,
    ]
    if reference:
        command.append("--reference")
    if checkpoint is not None:
        command += ["--checkpoint", str(checkpoint)]
    if save_at is not None:
        command += ["--save-at", str(save_at)]
    if resume_at:
        command += ["--resume-at", str(resume_at)]
    result = execute_subprocess_async(command, env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text(encoding="utf-8"))
