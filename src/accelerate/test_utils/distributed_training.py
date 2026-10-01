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

"""Launch training workers and read their results; scenarios own the assertions."""

import json
import os
import sys

from .testing import execute_subprocess_async, get_torch_dist_unique_port, path_in_accelerate_package


def run_training(
    output,
    *,
    batch_size,
    config_file=None,
    mixed_precision="no",
    gradient_accumulation_steps=1,
    launch_args=(),
    script="train_causal_lm.py",
    script_args=(),
):
    command = [sys.executable]
    if config_file is not None:
        command += [
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(config_file),
            "--mixed_precision",
            mixed_precision,
            "--main_process_port",
            str(get_torch_dist_unique_port()),
            *map(str, launch_args),
        ]
    command += [
        str(path_in_accelerate_package("test_utils", "scripts", "external_deps", script)),
        "--output",
        str(output),
        "--batch-size",
        str(batch_size),
        "--mixed-precision",
        mixed_precision,
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        *map(str, script_args),
    ]
    if config_file is None:
        command.append("--reference")

    result = execute_subprocess_async(command, env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text(encoding="utf-8"))
