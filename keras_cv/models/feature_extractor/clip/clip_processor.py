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

import tensorflow as tf
import tree

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import config
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.feature_extractor.clip.clip_processor_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_cv.models.feature_extractor.clip.clip_processor_utils import (
    convert_to_backend_tensor_or_python_list,
)
from keras_cv.models.feature_extractor.clip.clip_tokenizer import CLIPTokenizer

try:
    import keras_nlp
    from keras_nlp.layers import StartEndPacker
except ImportError:
    keras_nlp = None


@keras_cv_export("keras_cv.models.feature_extractor.CLIPProcessor")
class CLIPProcessor(keras.layers.Layer):
    """
    CLIPProcessor is a utility class that provides functionality for processing
    images and texts in the context of the CLIP (Contrastive Language-Image
    Pretraining) model.

    Args:
        input_resolution (int): The resolution of input images.
        vocabulary (str): string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string, it
            should be the file path to merge rules. The merge rule file should
            have one merge rule per line.

    Methods:
        process_images(image_path: List[str]): Transforms an image located at
            the specified path.

        process_texts(texts: Union[str, List[str]], context_length: int = 77):
            Processes a single text or a list of texts, returning packed token
            sequences.

    """

    def __init__(self, input_resolution, vocabulary, merges, **kwargs):
        super().__init__(**kwargs)
        if keras_nlp is None:
            raise ValueError(
                "ClipTokenizer requires keras-nlp. Please install "
                "using pip `pip install -U keras-nlp && pip install -U keras`"
            )
        self.input_resolution = input_resolution
        self.vocabulary = vocabulary
        self.merges = merges
        self.image_transform = self.transform_image
        self.tokenizer = CLIPTokenizer(
            vocabulary=self.vocabulary,
            merges=self.merges,
        )
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = StartEndPacker(
            start_value=self.tokenizer.token_to_id("<|startoftext|>"),
            end_value=self.tokenizer.token_to_id("<|endoftext|>"),
            pad_value=None,
            sequence_length=77,
            return_padding_mask=True,
        )
        self.built = True

    def transform_image(self, image_path):
        input_resolution = self.input_resolution
        mean = ops.array([0.48145466, 0.4578275, 0.40821073])
        std = ops.array([0.26862954, 0.26130258, 0.27577711])

        image = keras.utils.load_img(image_path)
        image = keras.utils.img_to_array(image)
        image = (
            ops.image.resize(
                image,
                (input_resolution, input_resolution),
                interpolation="bicubic",
            )
            / 255.0
        )
        central_fraction = input_resolution / image.shape[0]
        width, height = image.shape[0], image.shape[1]
        left = ops.cast((width - width * central_fraction) / 2, dtype="int32")
        top = ops.cast((height - height * central_fraction) / 2, dtype="int32")
        right = ops.cast((width + width * central_fraction) / 2, dtype="int32")
        bottom = ops.cast(
            (height + height * central_fraction) / 2, dtype="int32"
        )

        image = ops.slice(
            image, [left, top, 0], [right - left, bottom - top, 3]
        )

        image = (image - mean) / std
        return image

    def process_images(self, images):
        if isinstance(images, str):
            images = [images]

        def process_image(image):
            if isinstance(image, str):
                return self.image_transform(image)

        processed_images = list(map(process_image, images))
        processed_images = ops.stack(processed_images)
        return processed_images

    def _process_texts(self, texts, context_length: int = 77):
        # Ensure the layer is built
        if not self.built:
            self.build(None)

        texts = convert_inputs_to_list_of_tensor_segments(texts)

        if len(texts) != 1:
            raise ValueError(
                "CLIP requires each input feature to contain only "
                f"one segment, but received {len(texts)}."
            )

        token_ids, padding_mask = self.packer(
            self.tokenizer(texts[0]),
            sequence_length=context_length,
            add_start_value=True,
            add_end_value=True,
        )
        return {"token_ids": token_ids, "padding_mask": padding_mask}

    def call(self, texts, context_length: int = 77):
        return self._process_texts(texts, context_length=context_length)

    def get_build_config(self):
        return None

    def __call__(self, *args, **kwargs):
        # Always place on CPU for preprocessing, to avoid expensive back and
        # forth copies to GPU before the trainable model.
        with tf.device("cpu"):
            outputs = super().__call__(*args, **kwargs)

            # Jax and Torch lack native string and ragged types.
            # If we are running on those backends and not running with tf.data
            # (we are outside a tf.function), we covert all ragged and string
            # tensor to pythonic types.
            is_tf_backend = config.backend() == "tensorflow"
            is_in_tf_graph = not tf.executing_eagerly()
            if not is_tf_backend and not is_in_tf_graph:
                outputs = tree.map_structure(
                    convert_to_backend_tensor_or_python_list, outputs
                )

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_resolution": self.input_resolution,
                "vocabulary": self.vocabulary,
                "merges": self.merges,
            }
        )
        return config
