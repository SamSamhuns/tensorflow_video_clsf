from tensorflow.keras.applications import \
    InceptionV3, MobileNetV2, MobileNetV3Large, Xception, EfficientNetV2S, EfficientNetV2B3
from tensorflow.keras.applications.inception_v3 import \
    preprocess_input as preprocess_input_inception_v3
from tensorflow.keras.applications.mobilenet_v2 import \
    preprocess_input as preprocess_input_mobilenet_v2
from tensorflow.keras.applications.mobilenet_v3 import \
    preprocess_input as preprocess_input_mobilenet_v3
from tensorflow.keras.applications.xception import \
    preprocess_input as preprocess_input_xception
from tensorflow.keras.applications.efficientnet_v2 import \
    preprocess_input as preprocess_input_efficientnetV2

# 'efficientnetv2-b0': 224,
# 'efficientnetv2-b1': 240,
# 'efficientnetv2-b2': 260,
# 'efficientnetv2-b3': 300,
# 'efficientnetv2-s':  384,
# 'efficientnetv2-m':  480,
# 'efficientnetv2-l':  480,
# 'efficientnetv2-xl':  512,
# 'efficientnet_b0': 224,
# 'efficientnet_b1': 240,
# 'efficientnet_b2': 260,
# 'efficientnet_b3': 300,
# 'efficientnet_b4': 380,
# 'efficientnet_b5': 456,
# 'efficientnet_b6': 528,
# 'efficientnet_b7': 600,

IMAGE_SIZE = {"inception_v3": 299,
              "mobilenet_v2": 224,
              "mobilenet_v3_large": 224,
              "xception": 299,
              "efficientnet_v2b3": 300,
              "efficientnet_v2s": 384}

PREPROCESS_FUNCS = {"inception_v3": preprocess_input_inception_v3,
                    "mobilenet_v2": preprocess_input_mobilenet_v2,
                    "mobilenet_v3_large": preprocess_input_mobilenet_v3,
                    "xception": preprocess_input_xception,
                    "efficientnet_v2b3": preprocess_input_efficientnetV2,
                    "efficientnet_v2s": preprocess_input_efficientnetV2}

BACKBONE_MODELS = {"inception_v3": InceptionV3,
                   "mobilenet_v2": MobileNetV2,
                   "mobilenet_v3_large": MobileNetV3Large,
                   "xception": Xception,
                   "efficientnet_v2b3": EfficientNetV2B3,
                   "efficientnet_v2s": EfficientNetV2S}
