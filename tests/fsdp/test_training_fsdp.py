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

"""End-to-end training scenarios with explicit backend cases."""

from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from accelerate.test_utils.distributed_training import run_training
from accelerate.test_utils.testing import (
    execute_subprocess_async,
    get_launch_command,
    get_torch_dist_unique_port,
    require_cuda,
    require_huggingface_suite,
    require_multi_gpu,
)
from accelerate.utils import is_bf16_available, is_torch_version
from accelerate.utils.constants import FSDP2_PYTORCH_VERSION


FSDP_VERSIONS = [
    pytest.param(1, id="fsdp1"),
    pytest.param(
        2,
        id="fsdp2",
        marks=pytest.mark.skipif(
            not is_torch_version(">=", FSDP2_PYTORCH_VERSION),
            reason=f"FSDP2 requires torch {FSDP2_PYTORCH_VERSION} or later",
        ),
    ),
]


@pytest.mark.parametrize("fsdp_version", FSDP_VERSIONS)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training(tmp_path, fsdp_version):
    """Train in FP32 on eight examples per update: one GPU with eight, or two GPUs with four each.

    Compare the losses and require both runs to improve on the first batch.
    A plain-PyTorch reference can expose wrapper bugs that two Accelerate runs could share.
    """
    config_file = Path(__file__).parent / f"fsdp{fsdp_version}.yaml"
    expected_ranks = [{"backend": "FSDP", "world_size": 2, "fsdp_version": fsdp_version}] * 2

    reference = run_training(tmp_path / "reference.json", reference=True, batch_size=8, mixed_precision="no")
    trained = run_training(tmp_path / "trained.json", config_file=config_file, batch_size=4, mixed_precision="no")

    # Did the requested backend run on both GPUs for all ten updates?
    assert trained["ranks"] == expected_ranks
    assert len(reference["losses"]) == len(trained["losses"]) == 10

    # Did training match the reference? Training losses precede each update,
    # so compare the final measurement too.
    max_loss_difference = 1e-4
    assert_close(trained["losses"], reference["losses"], atol=max_loss_difference, rtol=0)
    assert_close(trained["final_loss"], reference["final_loss"], atol=max_loss_difference, rtol=0)

    # Agreement also accepts two runs that never learn; require progress on the first batch.
    min_loss_decrease = 1e-4
    assert reference["final_loss"] < reference["losses"][0] - min_loss_decrease
    assert trained["final_loss"] < trained["losses"][0] - min_loss_decrease


@pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")
@pytest.mark.parametrize("fsdp_version", FSDP_VERSIONS)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_mixed_precision(tmp_path, fsdp_version):
    """Does BF16 training complete with finite losses?

    Run ten updates on two GPUs. The worker also checks that a Linear layer produces BF16 output.
    """
    config_file = Path(__file__).parent / f"fsdp{fsdp_version}.yaml"
    expected_ranks = [{"backend": "FSDP", "world_size": 2, "fsdp_version": fsdp_version}] * 2

    trained = run_training(tmp_path / "bf16.json", config_file=config_file, batch_size=4, mixed_precision="bf16")

    # Did the requested backend run on both GPUs for all ten updates?
    assert trained["ranks"] == expected_ranks
    assert len(trained["losses"]) == 10

    # Training losses precede each update; check the final measurement too.
    assert torch.isfinite(torch.tensor(trained["losses"])).all()
    assert torch.isfinite(torch.tensor(trained["final_loss"]))


@pytest.mark.parametrize("fsdp_version", FSDP_VERSIONS)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_with_gradient_accumulation(tmp_path, fsdp_version):
    """Accumulating two smaller FP32 batches must match one larger batch.

    Keep eight examples per update across two GPUs. Compare four examples per GPU
    at once with two batches of two examples per GPU.
    """
    config_file = Path(__file__).parent / f"fsdp{fsdp_version}.yaml"
    expected_ranks = [{"backend": "FSDP", "world_size": 2, "fsdp_version": fsdp_version}] * 2

    large_batch = run_training(tmp_path / "large.json", config_file=config_file, batch_size=4, mixed_precision="no")
    accumulated = run_training(
        tmp_path / "accumulated.json",
        config_file=config_file,
        batch_size=2,
        mixed_precision="no",
        gradient_accumulation_steps=2,
    )

    # Did both batch sizes complete the same number of updates on the requested backend?
    assert large_batch["ranks"] == accumulated["ranks"] == expected_ranks
    assert len(large_batch["losses"]) == len(accumulated["losses"]) == 10

    # Did splitting each batch preserve the losses? Training losses precede each update,
    # so compare the final measurement too.
    assert_close(accumulated["losses"], large_batch["losses"], atol=1e-4, rtol=0)
    assert_close(accumulated["final_loss"], large_batch["final_loss"], atol=1e-4, rtol=0)


@pytest.mark.parametrize("fsdp_version", FSDP_VERSIONS)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_checkpoint_resume(tmp_path, fsdp_version):
    """Does restarting from a checkpoint reproduce uninterrupted training?

    Save after five updates, exit, and load in fresh processes for the remaining five.
    Compare the examples, learning rates and losses; momentum makes weights alone insufficient.
    """
    config_file = Path(__file__).parent / f"fsdp{fsdp_version}.yaml"
    expected_ranks = [{"backend": "FSDP", "world_size": 2, "fsdp_version": fsdp_version}] * 2
    checkpoint = tmp_path / "checkpoint"
    uninterrupted = run_training(
        tmp_path / "uninterrupted.json",
        config_file=config_file,
        batch_size=2,
        mixed_precision="no",
        gradient_accumulation_steps=2,
        script="resume_causal_lm.py",
    )
    partial = run_training(
        tmp_path / "partial.json",
        config_file=config_file,
        batch_size=2,
        mixed_precision="no",
        gradient_accumulation_steps=2,
        script="resume_causal_lm.py",
        script_args=["--save-checkpoint", checkpoint],
    )
    resumed = run_training(
        tmp_path / "resumed.json",
        config_file=config_file,
        batch_size=2,
        mixed_precision="no",
        gradient_accumulation_steps=2,
        script="resume_causal_lm.py",
        script_args=["--resume-from-checkpoint", checkpoint],
    )

    # Did both runs process the same examples, without repeating or missing an update?
    assert uninterrupted["ranks"] == partial["ranks"] == resumed["ranks"] == expected_ranks
    assert len(uninterrupted["losses"]) == 10
    assert len(partial["losses"]) == len(resumed["losses"]) == 5

    expected_sample_ids = [list(range(start, start + 8)) for start in range(0, 80, 8)]
    assert uninterrupted["sample_ids"] == expected_sample_ids
    assert partial["sample_ids"] + resumed["sample_ids"] == expected_sample_ids

    # Did the learning-rate schedule continue from the saved point?
    assert partial["learning_rates"] + resumed["learning_rates"] == uninterrupted["learning_rates"]

    # Did subsequent updates match? The first resumed loss alone cannot reveal lost momentum.
    max_loss_difference = 1e-5
    assert_close(partial["losses"] + resumed["losses"], uninterrupted["losses"], atol=max_loss_difference, rtol=0)
    assert_close(resumed["final_loss"], uninterrupted["final_loss"], atol=max_loss_difference, rtol=0)

    # Agreement also accepts two runs that never learn; require progress on the first batch.
    min_loss_decrease = 1e-4
    assert uninterrupted["final_loss"] < uninterrupted["losses"][0] - min_loss_decrease


@require_cuda
@require_multi_gpu
def test_fsdp1_checkpoint_with_empty_optimizer_and_extra_state(tmp_path):
    """Can FSDP1 restore both an empty optimizer state and dictionary model metadata?

    Save after a plain SGD update, then load the checkpoint in fresh processes.
    """
    command = get_launch_command(
        config_file=Path(__file__).parent / "fsdp1.yaml",
        fsdp_auto_wrap_policy="NO_WRAP",
        main_process_port=get_torch_dist_unique_port(),
    )
    command.append(Path(__file__).parent / "optimizer_checkpointing.py")

    # Both ranks check their restored optimizer and model metadata inside the worker.
    saved = execute_subprocess_async(command + ["--save-checkpoint", tmp_path])
    assert saved.returncode == 0, saved.stderr
    restored = execute_subprocess_async(command + ["--resume-from-checkpoint", tmp_path])
    assert restored.returncode == 0, restored.stderr
