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

import pytest
import torch

from accelerate.test_utils.sharded_training import run_training
from accelerate.utils import (
    is_bf16_available,
    is_datasets_available,
    is_deepspeed_available,
    is_transformers_available,
)


pytestmark = [
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs"),
    pytest.mark.skipif(not is_deepspeed_available(), reason="Requires DeepSpeed"),
    pytest.mark.skipif(
        not (is_transformers_available() and is_datasets_available()), reason="Requires transformers and datasets"
    ),
]
STAGES = [pytest.param(2, id="zero2"), pytest.param(3, id="zero3")]


def deepspeed_args(directory, stage, accumulation_steps=1):
    config = {
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": accumulation_steps,
        "gradient_clipping": 0,
        # Match PyTorch's initial scale; 2**32 can spend this short run only backing off.
        "fp16": {"enabled": "auto", "initial_scale_power": 16},
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": stage,
            # The default 500M-element buffers dwarf this tiny model.
            "reduce_bucket_size": 1_000_000,
            "allgather_bucket_size": 1_000_000,
            "stage3_prefetch_bucket_size": 1_000_000,
        },
    }
    config_file = directory / f"zero{stage}-accumulation{accumulation_steps}.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    return ["--use_deepspeed", f"--deepspeed_config_file={config_file}", "--zero3_init_flag=false"]


@pytest.mark.parametrize("stage", STAGES)
def test_training(tmp_path, stage):
    """Compare ZeRO with PyTorch on the same eight blocks per update."""
    reference = run_training(tmp_path / "reference.json", reference=True, batch_size=8)
    trained = run_training(tmp_path / "deepspeed.json", launch_args=deepspeed_args(tmp_path, stage))
    assert trained["backend"] == "DEEPSPEED"
    assert trained["world_size"] == 2
    assert len(trained["losses"]) == len(reference["losses"]) == 10
    torch.testing.assert_close(trained["losses"], reference["losses"], atol=1e-4, rtol=0)
    torch.testing.assert_close(trained["final_losses"], reference["final_losses"] * 2, atol=1e-4, rtol=0)
    assert max(trained["final_losses"]) < trained["losses"][0] - 0.01
    assert reference["final_losses"][0] < reference["losses"][0] - 0.01


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize(
    "precision",
    ["fp16", pytest.param("bf16", marks=pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16"))],
)
def test_training_mixed_precision(tmp_path, stage, precision):
    """Complete ten updates with DeepSpeed's requested compute dtype and lower the loss."""
    trained = run_training(
        tmp_path / "deepspeed.json", launch_args=deepspeed_args(tmp_path, stage), mixed_precision=precision
    )
    assert trained["backend"] == "DEEPSPEED"
    assert trained["world_size"] == 2
    assert len(trained["losses"]) == 10
    assert max(trained["final_losses"]) < trained["losses"][0] - 0.01
    torch.testing.assert_close(trained["final_losses"][0], trained["final_losses"][1], atol=1e-5, rtol=0)


@pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")
@pytest.mark.parametrize("stage", STAGES)
def test_training_with_gradient_accumulation(tmp_path, stage):
    """Compare complete windows while DeepSpeed owns loss scaling and optimizer steps."""
    large = run_training(tmp_path / "large.json", launch_args=deepspeed_args(tmp_path, stage), mixed_precision="bf16")
    accumulated = run_training(
        tmp_path / "accumulated.json",
        launch_args=deepspeed_args(tmp_path, stage, accumulation_steps=2),
        mixed_precision="bf16",
        batch_size=2,
        gradient_accumulation_steps=2,
    )
    assert large["backend"] == accumulated["backend"] == "DEEPSPEED"
    assert large["world_size"] == accumulated["world_size"] == 2
    assert len(large["losses"]) == len(accumulated["losses"]) == 10
    torch.testing.assert_close(accumulated["losses"], large["losses"], atol=1e-3, rtol=0)
    torch.testing.assert_close(accumulated["final_losses"], large["final_losses"], atol=1e-3, rtol=0)
    assert max(accumulated["final_losses"]) < accumulated["losses"][0] - 0.01


@pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")
@pytest.mark.parametrize("stage", STAGES)
def test_checkpoint_resume(tmp_path, stage):
    """Reload partitioned optimizer/model state in fresh processes and continue the same batches."""
    options = dict(
        launch_args=deepspeed_args(tmp_path, stage, accumulation_steps=2),
        mixed_precision="bf16",
        batch_size=2,
        gradient_accumulation_steps=2,
    )
    uninterrupted = run_training(tmp_path / "full.json", **options)
    partial = run_training(tmp_path / "partial.json", **options, checkpoint=tmp_path / "checkpoint", save_at=5)
    resumed = run_training(tmp_path / "resumed.json", **options, checkpoint=tmp_path / "checkpoint", resume_at=5)

    assert uninterrupted["backend"] == partial["backend"] == resumed["backend"] == "DEEPSPEED"
    assert uninterrupted["world_size"] == partial["world_size"] == resumed["world_size"] == 2
    assert len(uninterrupted["losses"]) == 10
    assert len(partial["losses"]) == len(resumed["losses"]) == 5
    torch.testing.assert_close(partial["losses"] + resumed["losses"], uninterrupted["losses"], atol=1e-5, rtol=0)
    torch.testing.assert_close(resumed["final_losses"], uninterrupted["final_losses"], atol=1e-5, rtol=0)
    assert partial["learning_rates"] + resumed["learning_rates"] == uninterrupted["learning_rates"]
    assert max(resumed["final_losses"]) < uninterrupted["losses"][0] - 0.01
