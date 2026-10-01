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

from pathlib import Path

import pytest
import torch

from accelerate.test_utils.distributed_training import run_training
from accelerate.test_utils.testing import (
    require_cuda,
    require_huggingface_suite,
    require_multi_gpu,
)
from accelerate.utils import is_bf16_available


@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training(tmp_path):
    """
    Compare DDP losses with single-GPU training on the same effective batches.
    A plain-PyTorch baseline can expose wrapper bugs that two Accelerate runs could share.
    """
    reference = run_training(tmp_path / "reference.json", reference=True, batch_size=8)
    distributed = run_training(tmp_path / "ddp.json", config_file=Path(__file__).with_name("ddp.yaml"), batch_size=4)

    loss_tolerance = 1e-4
    assert distributed["ranks"] == [{"backend": "MULTI_GPU", "world_size": 2}] * 2
    assert len(reference["losses"]) == len(distributed["losses"]) == 10
    torch.testing.assert_close(distributed["losses"], reference["losses"], atol=loss_tolerance, rtol=0)
    torch.testing.assert_close(distributed["final_loss"], reference["final_loss"], atol=loss_tolerance, rtol=0)

    # Agreement alone also accepts two runs that never learn. On the first batch,
    # require a loss decrease larger than the comparison tolerance.
    assert reference["final_loss"] < reference["losses"][0] - loss_tolerance
    assert distributed["final_loss"] < distributed["losses"][0] - loss_tolerance


@pytest.mark.parametrize(
    "mixed_precision, loss_tolerance",
    [
        pytest.param("fp16", 1e-4),
        pytest.param("bf16", 1e-3, marks=pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")),
    ],
)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_mixed_precision(tmp_path, mixed_precision, loss_tolerance):
    """Compare DDP with single-GPU training at the same requested precision."""
    reference = run_training(
        tmp_path / "reference.json", reference=True, batch_size=8, mixed_precision=mixed_precision
    )
    distributed = run_training(
        tmp_path / "ddp.json",
        config_file=Path(__file__).with_name("ddp.yaml"),
        batch_size=4,
        mixed_precision=mixed_precision,
    )

    assert distributed["ranks"] == [{"backend": "MULTI_GPU", "world_size": 2}] * 2
    assert len(reference["losses"]) == len(distributed["losses"]) == 10
    torch.testing.assert_close(distributed["losses"], reference["losses"], atol=loss_tolerance, rtol=0)
    torch.testing.assert_close(distributed["final_loss"], reference["final_loss"], atol=loss_tolerance, rtol=0)

    # Agreement alone also accepts two runs that never learn. On the first batch,
    # require a loss decrease larger than the comparison tolerance.
    assert reference["final_loss"] < reference["losses"][0] - loss_tolerance
    assert distributed["final_loss"] < distributed["losses"][0] - loss_tolerance


@pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_with_gradient_accumulation(tmp_path):
    """Keep BF16 and eight blocks per update: 2 ranks * 4 blocks, or 2 ranks * 2 blocks * 2 steps."""
    large_batch = run_training(
        tmp_path / "large.json", config_file=Path(__file__).with_name("ddp.yaml"), batch_size=4, mixed_precision="bf16"
    )
    accumulated = run_training(
        tmp_path / "accumulated.json",
        config_file=Path(__file__).with_name("ddp.yaml"),
        batch_size=2,
        mixed_precision="bf16",
        gradient_accumulation_steps=2,
    )

    loss_tolerance = 1e-3
    assert large_batch["ranks"] == accumulated["ranks"] == [{"backend": "MULTI_GPU", "world_size": 2}] * 2
    assert len(large_batch["losses"]) == len(accumulated["losses"]) == 10
    torch.testing.assert_close(accumulated["losses"], large_batch["losses"], atol=loss_tolerance, rtol=0)
    torch.testing.assert_close(accumulated["final_loss"], large_batch["final_loss"], atol=loss_tolerance, rtol=0)

    # Agreement alone also accepts two runs that never learn. On the first batch,
    # require a loss decrease larger than the comparison tolerance.
    assert large_batch["final_loss"] < large_batch["losses"][0] - loss_tolerance
    assert accumulated["final_loss"] < accumulated["losses"][0] - loss_tolerance


@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_checkpoint_resume(tmp_path):
    """Fresh processes must resume the same examples, momentum and learning-rate schedule."""
    options = dict(config_file=Path(__file__).with_name("ddp.yaml"))
    expected_ranks = [{"backend": "MULTI_GPU", "world_size": 2}] * 2

    options.update(script="resume_causal_lm.py", batch_size=2, gradient_accumulation_steps=2)
    checkpoint = tmp_path / "checkpoint"
    uninterrupted = run_training(tmp_path / "full.json", **options)
    partial = run_training(
        tmp_path / "partial.json", **options, script_args=["--checkpoint", checkpoint, "--save-at", "5"]
    )
    resumed = run_training(
        tmp_path / "resumed.json", **options, script_args=["--checkpoint", checkpoint, "--resume-at", "5"]
    )

    assert uninterrupted["ranks"] == partial["ranks"] == resumed["ranks"] == expected_ranks
    assert len(uninterrupted["losses"]) == 10
    assert len(partial["losses"]) == len(resumed["losses"]) == 5
    expected_ids = [list(range(start, start + 8)) for start in range(0, 80, 8)]
    assert uninterrupted["sample_ids"] == expected_ids
    assert partial["sample_ids"] + resumed["sample_ids"] == expected_ids
    assert uninterrupted["learning_rates"] == pytest.approx([0.1 * 0.95**step for step in range(10)])
    assert partial["learning_rates"] + resumed["learning_rates"] == uninterrupted["learning_rates"]
    torch.testing.assert_close(partial["losses"] + resumed["losses"], uninterrupted["losses"], atol=1e-5, rtol=0)
    torch.testing.assert_close(resumed["final_losses"], uninterrupted["final_losses"], atol=1e-5, rtol=0)
