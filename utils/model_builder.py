import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input, TimeDistributed, Dropout


class FeatureExtractor(tf.keras.Model):
    """
    Wrapper for tensorflow.keras.applications models
    if they do not have the compute_output_shape method implemented
    """

    def __init__(self, backbone_model, **kwargs):
        super(FeatureExtractor, self).__init__()
        self.feat_ext = backbone_model(**kwargs)
        self.feat_ext_output_shape = self.feat_ext.output_shape

    def build(self, input_shape):
        super(FeatureExtractor, self).build(input_shape)

    def call(self, x, **kwargs):
        input_shape = tf.shape(x)
        out_tensor = self.feat_ext(x)
        return tf.reshape(out_tensor, self.compute_output_shape(input_shape))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.feat_ext_output_shape[-1])


def build_video_clsf_model(model_name, backbone_model, MAX_FRAMES, img_size, gru_units, n_classes):
    cnn = FeatureExtractor(
        backbone_model,
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3))

    input_layer = Input((MAX_FRAMES, img_size, img_size, 3))
    model = TimeDistributed(cnn)(input_layer)
    model = GRU(gru_units)(model)
    model = Dense(gru_units // 2, activation="relu")(model)
    model = Dropout(0.2)(model)
    # import to set final dtype to float32 when using mixed_float16
    output = Dense(n_classes, activation="softmax",
                   dtype="float32")(model)

    model._name = model_name
    return Model([input_layer], output)


def build_video_clsf_masked_model(model_name, backbone_model, MAX_FRAMES, img_size, gru_units, n_classes):
    cnn = FeatureExtractor(
        backbone_model,
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3))

    image_input = Input((MAX_FRAMES, img_size, img_size, 3))
    mask_input = Input((MAX_FRAMES,), dtype="bool")
    model = TimeDistributed(cnn)(image_input)
    model = GRU(gru_units)(model, mask=mask_input)
    model = Dense(gru_units // 2, activation="relu")(model)
    model = Dropout(0.2)(model)
    # import to set final dtype to float32 when using mixed_float16
    output = Dense(n_classes, activation="softmax",
                   dtype="float32")(model)

    model._name = model_name
    return Model([image_input, mask_input], output)


def build_video_clsf_movinet_model(model_name, backbone_model, MAX_FRAMES, img_size, n_classes):
    cnn = backbone_model(num_frames=MAX_FRAMES,
                         img_size=img_size)

    x = cnn.layers[-1].output
    outputs = Dense(n_classes, activation="softmax", dtype="float32")(x)
    model = tf.keras.models.Model(inputs=cnn.input, outputs=outputs)

    model._name = model_name
    return model
