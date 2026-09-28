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

import json
import os
import sys
from pathlib import Path

import pytest
import torch

from accelerate.test_utils.testing import (
    execute_subprocess_async,
    get_torch_dist_unique_port,
    path_in_accelerate_package,
    require_cuda,
    require_huggingface_suite,
    require_multi_gpu,
)
from accelerate.utils import is_bf16_available, is_torch_version
from accelerate.utils.constants import FSDP2_PYTORCH_VERSION


FSDP_VARIANTS = [
    pytest.param("fsdp1.yaml", 1, id="fsdp1"),
    pytest.param(
        "fsdp2.yaml",
        2,
        id="fsdp2",
        marks=pytest.mark.skipif(
            not is_torch_version(">=", FSDP2_PYTORCH_VERSION),
            reason=f"FSDP2 requires torch {FSDP2_PYTORCH_VERSION} or later",
        ),
    ),
]


@pytest.mark.parametrize("config_name,fsdp_version", FSDP_VARIANTS)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_matches_pytorch(tmp_path, config_name, fsdp_version):
    reference = run_training(tmp_path / "reference.json", reference=True, batch_size=8)
    fsdp = run_training(tmp_path / "fsdp.json", config_name=config_name, batch_size=4)

    max_loss_difference = 1e-4
    min_loss_decrease = 1e-4
    assert_backend(fsdp, fsdp_version)
    assert fsdp["sample_ids"] == reference["sample_ids"]
    torch.testing.assert_close(fsdp["losses"], reference["losses"], atol=max_loss_difference, rtol=0)
    torch.testing.assert_close(fsdp["final_loss"], reference["final_loss"], atol=max_loss_difference, rtol=0)
    assert reference["final_loss"] < reference["losses"][0] - min_loss_decrease
    assert fsdp["final_loss"] < fsdp["losses"][0] - min_loss_decrease


@pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")
@pytest.mark.parametrize("config_name,fsdp_version", FSDP_VARIANTS)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_uses_mixed_precision(tmp_path, config_name, fsdp_version):
    fsdp = run_training(tmp_path / "fsdp.json", config_name=config_name, batch_size=4, mixed_precision="bf16")

    assert_backend(fsdp, fsdp_version)
    for rank in fsdp["ranks"]:
        assert rank["compute_dtypes"] == ["torch.bfloat16"]
    # FSDP2 can cast the returned loss to BF16 as part of its output policy.
    # This scenario checks finite training and actual BF16 compute, not FP32
    # scalar parity after that lossy cast. Ordinary training covers parity.
    assert len(fsdp["losses"]) == 6
    assert torch.isfinite(torch.tensor(fsdp["losses"])).all()
    assert torch.isfinite(torch.tensor(fsdp["final_loss"]))


@pytest.mark.parametrize("config_name,fsdp_version", FSDP_VARIANTS)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_with_gradient_accumulation(tmp_path, config_name, fsdp_version):
    large_batch = run_training(tmp_path / "large.json", config_name=config_name, batch_size=4)
    accumulated = run_training(
        tmp_path / "accumulated.json",
        config_name=config_name,
        batch_size=2,
        gradient_accumulation_steps=2,
    )

    max_loss_difference = 1e-4
    min_loss_decrease = 1e-4
    assert_backend(large_batch, fsdp_version)
    assert_backend(accumulated, fsdp_version)
    assert accumulated["sample_ids"] == large_batch["sample_ids"]
    torch.testing.assert_close(accumulated["losses"], large_batch["losses"], atol=max_loss_difference, rtol=0)
    torch.testing.assert_close(accumulated["final_loss"], large_batch["final_loss"], atol=max_loss_difference, rtol=0)
    assert large_batch["final_loss"] < large_batch["losses"][0] - min_loss_decrease
    assert accumulated["final_loss"] < accumulated["losses"][0] - min_loss_decrease


@pytest.mark.parametrize("config_name,fsdp_version", FSDP_VARIANTS)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_can_resume(tmp_path, config_name, fsdp_version):
    checkpoint = tmp_path / "checkpoint"
    uninterrupted = run_training(
        tmp_path / "uninterrupted.json",
        config_name=config_name,
        batch_size=4,
        optimizer="adamw",
        with_scheduler=True,
    )
    partial = run_training(
        tmp_path / "partial.json",
        config_name=config_name,
        batch_size=4,
        optimizer="adamw",
        with_scheduler=True,
        max_updates=3,
        checkpoint_dir=checkpoint,
    )
    resumed = run_training(
        tmp_path / "resumed.json",
        config_name=config_name,
        batch_size=4,
        optimizer="adamw",
        with_scheduler=True,
        resume_from_checkpoint=checkpoint,
        start_update=3,
    )

    assert_backend(resumed, fsdp_version)
    assert sum(rank["state"]["optimizer_state_entries"] for rank in partial["ranks"]) > 0
    for partial_rank, resumed_rank in zip(partial["ranks"], resumed["ranks"]):
        assert resumed_rank["loaded_state"] == partial_rank["state"]
    assert resumed["sample_ids"] == uninterrupted["sample_ids"][3:]
    assert resumed["learning_rates"] == uninterrupted["learning_rates"][3:]
    torch.testing.assert_close(resumed["losses"], uninterrupted["losses"][3:], atol=1e-6, rtol=0)
    torch.testing.assert_close(resumed["final_loss"], uninterrupted["final_loss"], atol=1e-6, rtol=0)


def assert_backend(result, fsdp_version):
    assert result["world_size"] == 2
    assert len(result["ranks"]) == 2
    assert {rank["distributed_type"] for rank in result["ranks"]} == {"FSDP"}
    assert {rank["fsdp_version"] for rank in result["ranks"]} == {fsdp_version}


def run_training(
    output,
    *,
    batch_size,
    config_name=None,
    mixed_precision="no",
    gradient_accumulation_steps=1,
    optimizer="sgd",
    with_scheduler=False,
    max_updates=6,
    checkpoint_dir=None,
    resume_from_checkpoint=None,
    start_update=0,
    reference=False,
):
    script = path_in_accelerate_package("test_utils", "scripts", "external_deps", "train_distributed_backend.py")
    command = [sys.executable]
    if not reference:
        command += [
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(Path(__file__).with_name(config_name)),
            "--main_process_port",
            str(get_torch_dist_unique_port()),
        ]

    command += [
        str(script),
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
        "--max-updates",
        str(max_updates),
        "--start-update",
        str(start_update),
    ]
    if reference:
        command.append("--reference")
    if with_scheduler:
        command.append("--with-scheduler")
    if checkpoint_dir:
        command += ["--checkpoint-dir", str(checkpoint_dir)]
    if resume_from_checkpoint:
        command += ["--resume-from-checkpoint", str(resume_from_checkpoint)]

    result = execute_subprocess_async(command, env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text(encoding="utf-8"))
