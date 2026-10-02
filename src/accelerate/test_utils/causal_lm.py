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

"""Fixtures and observations shared by causal-LM integration workers."""

import torch

from accelerate.utils import DistributedType


def register_mixed_precision_check(model, expected_dtype):
    """Check that a Linear layer produces output in the requested mixed precision.

    Training can succeed with similar losses even when mixed precision is
    inactive. Check the output dtype to ensure it was actually used.
    """

    def check_output_dtype(module, inputs, output):
        assert output.dtype == expected_dtype, f"Expected {expected_dtype} output, got {output.dtype}"

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(check_output_dtype)
            return

    raise ValueError("Expected a Linear layer to check mixed-precision output.")


def load_model_and_data():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # Training does not reuse the attention cache used for generation.
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32, use_cache=False)
    model.train()

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:100]")
    text = "\n\n".join(dataset["text"])
    token_ids = tokenizer(text, return_attention_mask=False)["input_ids"]

    block_size, num_blocks = 32, 80
    # Each block has 31 next-token targets. Equal-sized microbatches therefore
    # have equally weighted mean losses; eight blocks per update give ten updates.
    input_ids = torch.tensor(token_ids[: num_blocks * block_size]).reshape(num_blocks, block_size)
    return model, input_ids


def observe_backend(accelerator, model):
    """Report the prepared model's backend variant, rather than its requested config."""
    observation = {"backend": accelerator.distributed_type.value, "world_size": accelerator.num_processes}
    if accelerator.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp import FullyShardedDataParallel

        if isinstance(model, FullyShardedDataParallel):
            observation["fsdp_version"] = 1
        else:
            from torch.distributed.fsdp import FSDPModule

            assert isinstance(model, FSDPModule)
            observation["fsdp_version"] = 2
    elif accelerator.distributed_type == DistributedType.DEEPSPEED:
        observation["zero_stage"] = model.zero_optimization_stage()
    elif accelerator.distributed_type == DistributedType.MULTI_GPU:
        assert isinstance(model, torch.nn.parallel.DistributedDataParallel)
    return observation
