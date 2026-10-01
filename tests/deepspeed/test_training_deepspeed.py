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

from accelerate.test_utils.distributed_training import run_training
from accelerate.test_utils.testing import require_cuda, require_deepspeed, require_huggingface_suite, require_multi_gpu
from accelerate.utils import is_bf16_available


CASES = [pytest.param(1, id="zero1"), pytest.param(2, id="zero2"), pytest.param(3, id="zero3")]


@pytest.mark.parametrize("stage", CASES)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
@require_deepspeed
def test_training(tmp_path, stage):
    """Match plain PyTorch on the same ten FP32 effective batches."""
    options = dict(
        config_file=Path(__file__).with_name("training.yaml"),
        batch_size=4,
        launch_args=["--deepspeed_config_file", Path(__file__).with_name(f"training_zero{stage}.json")],
    )
    expected_ranks = [{"backend": "DEEPSPEED", "world_size": 2, "zero_stage": stage}] * 2

    reference = run_training(tmp_path / "reference.json", batch_size=8)
    trained = run_training(tmp_path / "trained.json", **options)

    assert trained["ranks"] == expected_ranks
    assert len(reference["losses"]) == len(trained["losses"]) == 10
    max_loss_difference = 1e-4
    min_loss_decrease = 1e-4
    torch.testing.assert_close(trained["losses"], reference["losses"], atol=max_loss_difference, rtol=0)
    torch.testing.assert_close(trained["final_loss"], reference["final_loss"], atol=max_loss_difference, rtol=0)
    assert reference["final_loss"] < reference["losses"][0] - min_loss_decrease
    assert trained["final_loss"] < trained["losses"][0] - min_loss_decrease


@pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")
@pytest.mark.parametrize("stage", CASES)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
@require_deepspeed
def test_training_mixed_precision(tmp_path, stage):
    """Complete finite BF16 training; the worker checks actual compute dtype."""
    options = dict(
        config_file=Path(__file__).with_name("training.yaml"),
        batch_size=4,
        launch_args=["--deepspeed_config_file", Path(__file__).with_name(f"training_zero{stage}.json")],
    )
    expected_ranks = [{"backend": "DEEPSPEED", "world_size": 2, "zero_stage": stage}] * 2

    trained = run_training(tmp_path / "bf16.json", **options, mixed_precision="bf16")

    assert trained["ranks"] == expected_ranks
    assert len(trained["losses"]) == 10
    assert torch.isfinite(torch.tensor(trained["losses"])).all()
    assert torch.isfinite(torch.tensor(trained["final_loss"]))


@pytest.mark.parametrize("stage", CASES)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
@require_deepspeed
def test_training_with_gradient_accumulation(tmp_path, stage):
    """Keep eight FP32 blocks per update: 2 ranks * 4, or 2 ranks * 2 * 2 steps."""
    options = dict(
        config_file=Path(__file__).with_name("training.yaml"),
        batch_size=4,
        launch_args=["--deepspeed_config_file", Path(__file__).with_name(f"training_zero{stage}.json")],
    )
    expected_ranks = [{"backend": "DEEPSPEED", "world_size": 2, "zero_stage": stage}] * 2

    large_batch = run_training(tmp_path / "large.json", **options)
    options.update(batch_size=2, gradient_accumulation_steps=2)
    accumulated = run_training(tmp_path / "accumulated.json", **options)

    assert large_batch["ranks"] == accumulated["ranks"] == expected_ranks
    assert len(large_batch["losses"]) == len(accumulated["losses"]) == 10
    torch.testing.assert_close(accumulated["losses"], large_batch["losses"], atol=1e-4, rtol=0)
    torch.testing.assert_close(accumulated["final_loss"], large_batch["final_loss"], atol=1e-4, rtol=0)


@pytest.mark.parametrize("stage", CASES)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
@require_deepspeed
def test_checkpoint_resume(tmp_path, stage):
    """Fresh processes must resume the same examples, momentum and learning-rate schedule."""
    options = dict(
        config_file=Path(__file__).with_name("training.yaml"),
        batch_size=4,
        launch_args=["--deepspeed_config_file", Path(__file__).with_name(f"training_zero{stage}.json")],
    )
    expected_ranks = [{"backend": "DEEPSPEED", "world_size": 2, "zero_stage": stage}] * 2

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
