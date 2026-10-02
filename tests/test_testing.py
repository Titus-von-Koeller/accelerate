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

import os
import signal
import sys

import pytest

from accelerate.test_utils.testing import execute_subprocess_async


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal exit codes")
def test_subprocess_signal_exit_raises():
    command = [
        sys.executable,
        "-c",
        "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
    ]
    with pytest.raises(RuntimeError, match=f"returncode {-signal.SIGTERM}"):
        execute_subprocess_async(command, echo=False)
