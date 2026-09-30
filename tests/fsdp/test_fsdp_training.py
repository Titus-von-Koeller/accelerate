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

import pytest
import torch

from accelerate.test_utils.sharded_training import run_training
from accelerate.utils import is_bf16_available, is_datasets_available, is_torch_version, is_transformers_available
from accelerate.utils.constants import FSDP2_PYTORCH_VERSION


pytestmark = [
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs"),
    pytest.mark.skipif(
        not (is_transformers_available() and is_datasets_available()), reason="Requires transformers and datasets"
    ),
]
FSDP2 = pytest.mark.skipif(not is_torch_version(">=", FSDP2_PYTORCH_VERSION), reason="Requires FSDP2")
VERSIONS = [pytest.param(1, id="fsdp1"), pytest.param(2, marks=FSDP2, id="fsdp2")]


def fsdp_args(version, *, reshard=True, state_dict_type="SHARDED_STATE_DICT"):
    args = [
        "--use_fsdp",
        f"--fsdp_version={version}",
        "--fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP",
        "--fsdp_transformer_layer_cls_to_wrap=Qwen2DecoderLayer",
        "--fsdp_cpu_ram_efficient_loading=false",
        "--fsdp_offload_params=false",
        f"--fsdp_state_dict_type={state_dict_type}",
    ]
    if version == 1:
        args += [
            "--fsdp_use_orig_params=true",
            "--fsdp_sync_module_states=true",
            f"--fsdp_sharding_strategy={'FULL_SHARD' if reshard else 'SHARD_GRAD_OP'}",
        ]
    else:
        args += [f"--fsdp_reshard_after_forward={str(reshard).lower()}"]
    return args


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("reshard", [True, False], ids=["reshard", "keep_after_forward"])
def test_training(tmp_path, version, reshard):
    """Compare sharded training with PyTorch on the same eight blocks per update."""
    reference = run_training(tmp_path / "reference.json", reference=True, batch_size=8)
    trained = run_training(tmp_path / "fsdp.json", launch_args=fsdp_args(version, reshard=reshard))

    assert trained["backend"] == "FSDP"
    assert trained["world_size"] == 2
    assert len(trained["losses"]) == len(reference["losses"]) == 10
    torch.testing.assert_close(trained["losses"], reference["losses"], atol=1e-4, rtol=0)
    torch.testing.assert_close(trained["final_losses"], reference["final_losses"] * 2, atol=1e-4, rtol=0)
    # Learning is a separate requirement from numerical agreement.
    assert max(trained["final_losses"]) < trained["losses"][0] - 0.01
    assert reference["final_losses"][0] < reference["losses"][0] - 0.01


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(
    "precision",
    ["fp16", pytest.param("bf16", marks=pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16"))],
)
def test_training_mixed_precision(tmp_path, version, precision):
    """Complete ten updates using the requested compute dtype and lower the loss."""
    trained = run_training(tmp_path / "fsdp.json", launch_args=fsdp_args(version), mixed_precision=precision)
    assert trained["backend"] == "FSDP"
    assert trained["world_size"] == 2
    assert len(trained["losses"]) == 10
    assert max(trained["final_losses"]) < trained["losses"][0] - 0.01
    torch.testing.assert_close(trained["final_losses"][0], trained["final_losses"][1], atol=1e-5, rtol=0)


@pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")
@pytest.mark.parametrize("version", VERSIONS)
def test_training_with_gradient_accumulation(tmp_path, version):
    """Keep eight blocks per update: 2 ranks * 4 blocks or 2 ranks * 2 blocks * 2 steps."""
    args = fsdp_args(version)
    large = run_training(tmp_path / "large.json", launch_args=args, mixed_precision="bf16")
    accumulated = run_training(
        tmp_path / "accumulated.json",
        launch_args=args,
        mixed_precision="bf16",
        batch_size=2,
        gradient_accumulation_steps=2,
    )
    assert large["backend"] == accumulated["backend"] == "FSDP"
    assert large["world_size"] == accumulated["world_size"] == 2
    assert len(large["losses"]) == len(accumulated["losses"]) == 10
    torch.testing.assert_close(accumulated["losses"], large["losses"], atol=1e-3, rtol=0)
    torch.testing.assert_close(accumulated["final_losses"], large["final_losses"], atol=1e-3, rtol=0)
    assert max(accumulated["final_losses"]) < accumulated["losses"][0] - 0.01


@pytest.mark.parametrize(
    "version, state_dict_type, optimizer, with_extra_state",
    [
        pytest.param(1, "FULL_STATE_DICT", "sgd", False, id="fsdp1-full"),
        pytest.param(1, "SHARDED_STATE_DICT", "sgd", False, id="fsdp1-sharded-momentum"),
        pytest.param(1, "SHARDED_STATE_DICT", "sgd", True, id="fsdp1-sharded-extra-state"),
        pytest.param(1, "SHARDED_STATE_DICT", "sgd_plain", False, id="fsdp1-sharded-plain"),
        pytest.param(1, "SHARDED_STATE_DICT", "adamw", False, id="fsdp1-sharded-adamw"),
        pytest.param(2, "FULL_STATE_DICT", "sgd", False, marks=FSDP2, id="fsdp2-full"),
        pytest.param(2, "SHARDED_STATE_DICT", "sgd", False, marks=FSDP2, id="fsdp2-sharded"),
    ],
)
def test_checkpoint_resume(tmp_path, version, state_dict_type, optimizer, with_extra_state):
    """Resume in fresh processes at an update boundary with optimizer and scheduler state."""
    args = fsdp_args(version, state_dict_type=state_dict_type)
    options = dict(
        launch_args=args,
        batch_size=2,
        gradient_accumulation_steps=2,
        optimizer=optimizer,
        with_extra_state=with_extra_state,
    )
    uninterrupted = run_training(tmp_path / "full.json", **options)
    partial = run_training(tmp_path / "partial.json", **options, checkpoint=tmp_path / "checkpoint", save_at=5)
    resumed = run_training(tmp_path / "resumed.json", **options, checkpoint=tmp_path / "checkpoint", resume_at=5)

    assert uninterrupted["backend"] == partial["backend"] == resumed["backend"] == "FSDP"
    assert uninterrupted["world_size"] == partial["world_size"] == resumed["world_size"] == 2
    if with_extra_state:
        assert partial["extra_state_revision"] == resumed["extra_state_revision"] == 5
    assert len(uninterrupted["losses"]) == 10
    assert len(partial["losses"]) == len(resumed["losses"]) == 5
    torch.testing.assert_close(partial["losses"] + resumed["losses"], uninterrupted["losses"], atol=1e-5, rtol=0)
    torch.testing.assert_close(resumed["final_losses"], uninterrupted["final_losses"], atol=1e-5, rtol=0)
    assert partial["learning_rates"] + resumed["learning_rates"] == uninterrupted["learning_rates"]
    assert max(resumed["final_losses"]) < uninterrupted["losses"][0] - 0.01
