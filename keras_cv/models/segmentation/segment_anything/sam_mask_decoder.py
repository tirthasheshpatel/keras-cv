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

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.layers.detectron2_layers import MLP
from keras_cv.layers.serializable_sequential import SerializableSequential
from keras_cv.models.segmentation.segment_anything.sam_transformer import (
    TwoWayTransformer,
)


@keras_cv_export("keras_cv.layers.SAMMaskDecoder")
class SAMMaskDecoder(keras.layers.Layer):
    """Mask decoder for the Segment Anything Model (SAM).

    This lightweight module efficiently maps the image embedding and a set of
    prompt embeddings to an output mask. Before applying the transformer
    decoder, the layer first inserts into the set of prompt embeddings a
    learned output token embedding that will be used at the decoder's output.
    For simplicity, these embeddings (not including the image embedding) are
    collectively called "tokens".

    The image embeddings, positional image embeddings, and tokens are passed
    through a transformer decoder. After running the decoder, the layer
    upsamples the updated image embedding by 4x with two transposed
    convolutional layers (now it's downscaled 4x relative to the input
    image). Then, the tokens attend once more to the image embedding and
    the updated output token embedding are passed to a small 3-layer MLP that
    outputs a vector matching the channel dimension of the upscaled image
    embedding. Finally, a mask is predicted with a spatially point-wise
    product between the upscaled image embedding and the MLP's output.

    Args:
        transformer_dim (int, optional): The number of input features to the
            transformer decoder. Defaults to `256`.
        transformer (keras.layers.Layer, optional): A transformer decoder.
            Defaults to `None`. When `None`, a
            `keras_cv.models.TwoWayTransformer` layer is used.
        num_multimask_outputs (int, optional): Number of multimask outputs.
            The model would generate these many extra masks. The total masks
            generated by the model are `1 + num_multimask_outputs`. Defaults
            to `3`.
        iou_head_depth (int, optional): The depth of the dense net used to
            predict the IoU confidence score. Defaults to `3`.
        iou_head_hidden_dim (int, optional): The number of units in the hidden
            layers used in the dense net to predict the IoU confidence score.
            Defaults to `256`.
        activation (str, optional): Activation to use in the mask upscaler
            network. Defaults to `"gelu"`.

    References:
        - [Segment Anything](https://arxiv.org/abs/2304.02643)
    """

    def __init__(
        self,
        *,
        transformer_dim=256,
        transformer=None,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        activation="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transformer_dim = transformer_dim
        if transformer is None:
            transformer = TwoWayTransformer()
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.activation = activation

        self.iou_token = keras.layers.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = keras.layers.Embedding(
            self.num_mask_tokens, transformer_dim
        )

        self.output_upscaling = SerializableSequential(
            [
                keras.layers.Conv2DTranspose(
                    transformer_dim // 4, kernel_size=2, strides=2
                ),
                keras.layers.LayerNormalization(epsilon=1e-6),
                keras.layers.Activation(activation),
                keras.layers.Conv2DTranspose(
                    transformer_dim // 8, kernel_size=2, strides=2
                ),
                keras.layers.Activation(activation),
            ]
        )

        self.output_hypernetworks_mlps = [
            MLP(transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ]

        self.iou_prediction_head = MLP(
            iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def build(self, input_shape=None):
        self.transformer.build()
        self.iou_token.build([None])
        self.mask_tokens.build([None])
        self.output_upscaling.build([None, None, None, self.transformer_dim])
        for mlp in self.output_hypernetworks_mlps:
            mlp.build([None, self.transformer_dim])
        self.iou_prediction_head.build([None, self.transformer_dim])
        self.built = True

    def call(self, inputs):
        image_embeddings = inputs["image_embeddings"]
        image_pe = inputs["image_pe"]
        sparse_prompt_embeddings = inputs["sparse_prompt_embeddings"]
        dense_prompt_embeddings = inputs["dense_prompt_embeddings"]

        masks, iou_pred = self._predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        return {"masks": masks, "iou_pred": iou_pred}

    def _predict_masks(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
    ):
        indices_iou = ops.arange(1, dtype="int32")
        indices_mask = ops.arange(self.num_mask_tokens, dtype="int32")

        output_tokens = ops.concatenate(
            [self.iou_token(indices_iou), self.mask_tokens(indices_mask)],
            axis=0,
        )
        output_tokens = ops.broadcast_to(
            output_tokens[None, ...],
            shape=(
                ops.shape(sparse_prompt_embeddings)[0],
                ops.shape(output_tokens)[0],
                ops.shape(output_tokens)[1],
            ),
        )
        tokens = ops.concatenate(
            [output_tokens, sparse_prompt_embeddings], axis=1
        )

        source = ops.broadcast_to(
            image_embeddings,
            shape=(
                ops.shape(tokens)[0],
                ops.shape(image_embeddings)[1],
                ops.shape(image_embeddings)[2],
                ops.shape(image_embeddings)[3],
            ),
        )
        source = source + dense_prompt_embeddings
        positional_source = ops.broadcast_to(
            image_pe,
            shape=(
                ops.shape(tokens)[0],
                ops.shape(image_embeddings)[1],
                ops.shape(image_embeddings)[2],
                ops.shape(image_embeddings)[3],
            ),
        )
        shape = ops.shape(source)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]

        hidden_state, source = self.transformer(
            source, positional_source, tokens
        )
        iou_token_out = hidden_state[:, 0, :]
        mask_tokens_out = hidden_state[:, 1 : (1 + self.num_mask_tokens), :]

        source = ops.reshape(source, (B, H, W, C))
        upscaled_embeddings = self.output_upscaling(source)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = ops.stack(hyper_in_list, axis=1)
        shape = ops.shape(upscaled_embeddings)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        upscaled_embeddings = ops.reshape(
            ops.transpose(upscaled_embeddings, axes=(0, 3, 1, 2)),
            (B, C, H * W),
        )
        masks = ops.reshape(
            hyper_in @ upscaled_embeddings, (B, self.num_mask_tokens, H, W)
        )

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "transformer_dim": self.transformer_dim,
                "transformer": keras.saving.serialize_keras_object(
                    self.transformer
                ),
                "num_multimask_outputs": self.num_multimask_outputs,
                "iou_head_depth": self.iou_head_depth,
                "iou_head_hidden_dim": self.iou_head_hidden_dim,
                "activation": self.activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config.update(
            {"transformer": keras.layers.deserialize(config["transformer"])}
        )
        return super().from_config(config)
