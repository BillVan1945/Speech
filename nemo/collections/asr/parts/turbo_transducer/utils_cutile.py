# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import cuda.tile as ct

def swizzle_2d_as_triton(i, j, size_i, size_j, size_g):
    """
    Swizzle 2d, following Triton interface https://triton-lang.org/main/python-api/generated/triton.language.swizzle2d.html
    """
    # TODO: fix to make swizzle
    i_new, j_new = i, j
    return i_new, j_new
