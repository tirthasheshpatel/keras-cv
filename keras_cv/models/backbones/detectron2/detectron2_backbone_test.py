# Copyright 2023 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import pytest

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.backbones.detectron2.detectron2_aliases import (
    SAMViTDetBBackbone,
)
from keras_cv.tests.test_case import TestCase


@pytest.mark.extra_large
class TestViTDetBackbone(TestCase):
    def test_call_and_save(self):
        model = SAMViTDetBBackbone()
        x = np.ones((1, 1024, 1024, 3))
        x_out = ops.convert_to_numpy(model(x))
        num_parameters = sum(
            np.prod(tuple(x.shape)) for x in model.trainable_variables
        )
        self.assertEqual(x_out.shape, (1, 64, 64, 256))
        self.assertEqual(num_parameters, 89_670_912)

        # saving test
        path = os.path.join(self.get_temp_dir(), "sam_tf_model.keras")
        model.save(path)
        loaded_model = keras.saving.load_model(path)
        x_out_loaded = ops.convert_to_numpy(loaded_model(x))
        self.assertAllClose(x_out, x_out_loaded)