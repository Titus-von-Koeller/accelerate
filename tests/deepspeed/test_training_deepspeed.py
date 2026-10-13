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
from accelerate.test_utils.testing import require_cuda, require_deepspeed, require_huggingface_suite, require_multi_gpu
from accelerate.utils import is_bf16_available


LAUNCH_CONFIG_FILE = Path(__file__).parent / "training.yaml"
ZERO_STAGES = [pytest.param(1, id="zero1"), pytest.param(2, id="zero2"), pytest.param(3, id="zero3")]


@pytest.mark.parametrize("stage", ZERO_STAGES)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
@require_deepspeed
def test_training(tmp_path, stage):
    """Train in FP32 on eight examples per update: one GPU with eight, or two GPUs with four each.

    Compare the losses and require both runs to improve on the first batch.
    A plain-PyTorch reference can expose wrapper bugs that two Accelerate runs could share.
    """
    deepspeed_config_file = Path(__file__).parent / f"training_zero{stage}.json"
    expected_ranks = [{"backend": "DEEPSPEED", "world_size": 2, "zero_stage": stage}] * 2

    reference = run_training(tmp_path / "reference.json", reference=True, batch_size=8, mixed_precision="no")
    trained = run_training(
        tmp_path / "trained.json",
        config_file=LAUNCH_CONFIG_FILE,
        deepspeed_config_file=deepspeed_config_file,
        batch_size=4,
        mixed_precision="no",
    )

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
@pytest.mark.parametrize("stage", ZERO_STAGES)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
@require_deepspeed
def test_training_mixed_precision(tmp_path, stage):
    """Does BF16 training complete with finite losses?

    Run ten updates on two GPUs. The worker also checks that a Linear layer produces BF16 output.
    """
    deepspeed_config_file = Path(__file__).parent / f"training_zero{stage}.json"
    expected_ranks = [{"backend": "DEEPSPEED", "world_size": 2, "zero_stage": stage}] * 2

    trained = run_training(
        tmp_path / "bf16.json",
        config_file=LAUNCH_CONFIG_FILE,
        deepspeed_config_file=deepspeed_config_file,
        batch_size=4,
        mixed_precision="bf16",
    )

    # Did the requested backend run on both GPUs for all ten updates?
    assert trained["ranks"] == expected_ranks
    assert len(trained["losses"]) == 10

    # Training losses precede each update; check the final measurement too.
    assert torch.isfinite(torch.tensor(trained["losses"])).all()
    assert torch.isfinite(torch.tensor(trained["final_loss"]))


@pytest.mark.parametrize("stage", ZERO_STAGES)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
@require_deepspeed
def test_training_with_gradient_accumulation(tmp_path, stage):
    """Accumulating two smaller FP32 batches must match one larger batch.

    Keep eight examples per update across two GPUs. Compare four examples per GPU
    at once with two batches of two examples per GPU.
    """
    deepspeed_config_file = Path(__file__).parent / f"training_zero{stage}.json"
    expected_ranks = [{"backend": "DEEPSPEED", "world_size": 2, "zero_stage": stage}] * 2

    large_batch = run_training(
        tmp_path / "large.json",
        config_file=LAUNCH_CONFIG_FILE,
        deepspeed_config_file=deepspeed_config_file,
        batch_size=4,
        mixed_precision="no",
    )
    accumulated = run_training(
        tmp_path / "accumulated.json",
        config_file=LAUNCH_CONFIG_FILE,
        deepspeed_config_file=deepspeed_config_file,
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


@pytest.mark.parametrize("stage", ZERO_STAGES)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
@require_deepspeed
def test_checkpoint_resume(tmp_path, stage):
    """Does restarting from a checkpoint reproduce uninterrupted training?

    Save after five updates, exit, and load in fresh processes for the remaining five.
    Compare the examples, learning rates and losses; momentum makes weights alone insufficient.
    """
    deepspeed_config_file = Path(__file__).parent / f"training_zero{stage}.json"
    expected_ranks = [{"backend": "DEEPSPEED", "world_size": 2, "zero_stage": stage}] * 2
    checkpoint = tmp_path / "checkpoint"
    uninterrupted = run_training(
        tmp_path / "uninterrupted.json",
        config_file=LAUNCH_CONFIG_FILE,
        deepspeed_config_file=deepspeed_config_file,
        batch_size=2,
        mixed_precision="no",
        gradient_accumulation_steps=2,
        script="resume_causal_lm.py",
    )
    partial = run_training(
        tmp_path / "partial.json",
        config_file=LAUNCH_CONFIG_FILE,
        deepspeed_config_file=deepspeed_config_file,
        batch_size=2,
        mixed_precision="no",
        gradient_accumulation_steps=2,
        script="resume_causal_lm.py",
        script_args=["--save-checkpoint", checkpoint],
    )
    resumed = run_training(
        tmp_path / "resumed.json",
        config_file=LAUNCH_CONFIG_FILE,
        deepspeed_config_file=deepspeed_config_file,
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


@pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")
@require_cuda
@require_multi_gpu
@require_huggingface_suite
@require_deepspeed
def test_zero3_mixed_precision_with_gradient_accumulation(tmp_path):
    """Does BF16 training complete when ZeRO3 also accumulates gradients?

    Accumulate two batches of two examples per GPU, keeping eight examples per update.
    Require finite losses; the worker also checks that a Linear layer produces BF16 output.
    """
    trained = run_training(
        tmp_path / "bf16_accumulated.json",
        config_file=LAUNCH_CONFIG_FILE,
        deepspeed_config_file=Path(__file__).parent / "training_zero3.json",
        batch_size=2,
        mixed_precision="bf16",
        gradient_accumulation_steps=2,
    )

    # Did ZeRO3 run on both GPUs for all ten updates?
    assert trained["ranks"] == [{"backend": "DEEPSPEED", "world_size": 2, "zero_stage": 3}] * 2
    assert len(trained["losses"]) == 10

    # Training losses precede each update; check the final measurement too.
    assert torch.isfinite(torch.tensor(trained["losses"])).all()
    assert torch.isfinite(torch.tensor(trained["final_loss"]))
