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

"""Model and data shared by causal-LM training and checkpoint tests."""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_data():
    """Load the tiny checkpoint and 80 equal-length text blocks in a fixed order."""
    checkpoint = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # Training does not reuse the attention cache used for generation.
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        use_cache=False,
    )
    model.train()

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:100]")
    text = "\n\n".join(dataset["text"])
    token_ids = tokenizer(text, return_attention_mask=False)["input_ids"]

    block_size, num_blocks = 32, 80
    # Each block has 31 next-token targets, so equally sized microbatch losses
    # can be averaged without reweighting. Eight blocks per update give ten complete updates.
    input_ids = torch.tensor(token_ids[: num_blocks * block_size]).reshape(num_blocks, block_size)
    return model, input_ids
