# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from dataclasses import dataclass

import pytest
import torch
from omegaconf import OmegaConf

from nemo.core.config import CosineAnnealingParams, InverseSquareRootAnnealingParams, SchedulerParams, SGDParams
from nemo.core.config.modelPT import ModelConfig, ModelConfigBuilder
from nemo.core.optim.lr_scheduler import CosineAnnealing, prepare_lr_scheduler


def _sgd(lr: float) -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=lr)


class TestModelConfigBuilder:
    @pytest.mark.unit
    def test_set_optim_scheduler_config(self):
        # the built scheduler config is the parameter config plus the scheduler's name, nothing else
        builder = ModelConfigBuilder(ModelConfig())
        sched_params = InverseSquareRootAnnealingParams(max_steps=20, warmup_steps=5)
        builder.set_optim(SGDParams(lr=1.0), sched_params)
        assert vars(builder.model_cfg.optim.sched) == {**vars(sched_params), 'name': 'InverseSquareRootAnnealing'}

        # `min_lr` stays where the parameter config has it, through the structured config that ModelPT builds
        builder.set_optim(SGDParams(lr=1.0), CosineAnnealingParams(max_steps=20, min_lr=0.02))
        cfg = OmegaConf.structured(builder.model_cfg)
        assert cfg.optim.sched.min_lr == 0.02
        scheduler = prepare_lr_scheduler(_sgd(cfg.optim.lr), cfg.optim.sched)['scheduler']
        assert isinstance(scheduler, CosineAnnealing)
        assert scheduler.min_lr == 0.02

    @pytest.mark.unit
    def test_set_optim_scheduler_without_min_lr(self):
        # a scheduler that takes no `min_lr`, here torch's StepLR through its own parameter config
        @dataclass
        class StepLRParams(SchedulerParams):
            max_steps: int = 20
            step_size: int = 2
            gamma: float = 0.5

        builder = ModelConfigBuilder(ModelConfig())
        builder.set_optim(SGDParams(lr=1.0), StepLRParams())
        cfg = OmegaConf.structured(builder.model_cfg)
        scheduler = prepare_lr_scheduler(_sgd(cfg.optim.lr), cfg.optim.sched)['scheduler']
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 2
        assert scheduler.gamma == 0.5
