# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
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

"""D4RL Hopper dataset from Mujoco."""

from typing import Any

from tensorflow_datasets.d4rl import mujoco_dataset
import tensorflow_datasets.public_api as tfds


class D4rlMujocoHopper(mujoco_dataset.D4RLMujocoDatasetBuilder):
  """DatasetBuilder for hopper dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = mujoco_dataset.BUILDER_CONFIGS

  def __init__(self, **kwargs: Any):
    config = mujoco_dataset.DatasetConfig(
        name='hopper', obs_len=11, action_len=3, qpos_len=6, qvel_len=6)
    super().__init__(ds_config=config, **kwargs)

