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

"""RefCoco datasets tests."""

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.vision_language.refcoco import refcoco


class RefcocoGoogleTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for refcoco/google."""

  DATASET_CLASS = refcoco.RefCoco
  BUILDER_CONFIG_NAMES_TO_TEST = ['refcoco/google']
  SPLITS = {
      tfds.Split.TRAIN: 1,
      tfds.Split.VALIDATION: 2,
      tfds.Split.TEST: 1,
  }


class RefcocoUncTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for refcoco/unc."""

  DATASET_CLASS = refcoco.RefCoco
  BUILDER_CONFIG_NAMES_TO_TEST = ['refcoco/unc']
  SPLITS = {
      tfds.Split.TRAIN: 1,
      tfds.Split.VALIDATION: 1,
      'testA': 1,
      'testB': 1,
  }


class RefcocoplusUncTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for refcoco+/unc."""

  DATASET_CLASS = refcoco.RefCoco
  BUILDER_CONFIG_NAMES_TO_TEST = ['refcoco+/unc']
  SPLITS = {
      tfds.Split.TRAIN: 1,
      tfds.Split.VALIDATION: 1,
      'testA': 1,
      'testB': 1,
  }


class RefcocogGoogleTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for refcocog/google."""

  DATASET_CLASS = refcoco.RefCoco
  BUILDER_CONFIG_NAMES_TO_TEST = ['refcocog/google']
  SPLITS = {
      tfds.Split.TRAIN: 1,
      tfds.Split.VALIDATION: 2,
  }


class RefcocogUmdTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for refcocog/umd."""

  DATASET_CLASS = refcoco.RefCoco
  BUILDER_CONFIG_NAMES_TO_TEST = ['refcocog/umd']
  SPLITS = {
      tfds.Split.TRAIN: 1,
      tfds.Split.VALIDATION: 1,
      tfds.Split.TEST: 1,
  }


if __name__ == '__main__':
  tfds.testing.test_main()
