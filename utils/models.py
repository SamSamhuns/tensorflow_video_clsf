from functools import partial

# try importing efficient-net v2s
try:
    from tensorflow.keras.applications import \
        EfficientNetV2S, EfficientNetV2B3
    from tensorflow.keras.applications.efficientnet_v2 import \
        preprocess_input as preprocess_input_efficientnetV2
except ImportError as e:
    print(e)
    EfficientNetV2S, EfficientNetV2B3 = None, None
    preprocess_input_efficientnetV2 = None
# try import custom places265 resnet model
try:
    from .places365.resnet50 import \
        load_resnet50_places365_model as ResNet50_Places365
    from .places365.resnet50 import \
        preprocess_input as preprocess_input_resnet50_places365
except ImportError as e:
    print(e)
    ResNet50_Places365 = None
    preprocess_input_resnet50_places365 = None
# try import custom movinet models
try:
    from .movinet.movinet import \
        load_movinet_model as Movinet
    from .movinet.movinet import \
        preprocess_input as preprocess_input_movinet
except ImportError as e:
    print(e)
    Movinet = None
    preprocess_input_movinet = None
# try import custom convnext models
try:
    from .convnext.convnext import \
        load_convnext_model as ConvNext
    from .convnext.convnext import \
        preprocess_input as preprocess_input_convnext
except ImportError as e:
    print(e)
    ConvNext = None
    preprocess_input_convnext = None

# built-in applications models
from tensorflow.keras.applications import \
    InceptionV3, MobileNetV2, MobileNetV3Large, Xception

from tensorflow.keras.applications.inception_v3 import \
    preprocess_input as preprocess_input_inception_v3
from tensorflow.keras.applications.mobilenet_v2 import \
    preprocess_input as preprocess_input_mobilenet_v2
from tensorflow.keras.applications.mobilenet_v3 import \
    preprocess_input as preprocess_input_mobilenet_v3
from tensorflow.keras.applications.xception import \
    preprocess_input as preprocess_input_xception

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
# movinet kinetics-600
# Model Name	    Top-1 Acc Top-5 Acc	Input Shape	     GFLOPs FPS     URL(prefix=https://tfhub.dev/tensorflow/movinet/)
# MoViNet-A0-Base	72.28	  90.92	    50 x 172 x 172	 2.7    5 fps   a0/base/kinetics-600/classification/3
# MoViNet-A1-Base	76.69	  93.40	    50 x 172 x 172	 6.0    5 fps   a1/base/kinetics-600/classification/3
# MoViNet-A2-Base	78.62	  94.17	    50 x 224 x 224	 10     5 fps   a2/base/kinetics-600/classification/3
# MoViNet-A3-Base	81.79	  95.67	    120 x 256 x 256	 57     12 fps  a3/base/kinetics-600/classification/3
# MoViNet-A4-Base	83.48	  96.16	    80 x 290 x 290	 110    8 fps   a4/base/kinetics-600/classification/3
# MoViNet-A5-Base	84.27	  96.39	    120 x 320 x 320	 280    12 fps  a5/base/kinetics-600/classification/3

IMAGE_SIZE = {"inception_v3": 299,
              "mobilenet_v2": 224,
              "mobilenet_v3_large": 224,
              "xception": 299,
              "efficientnet_v2b3": 300,
              "efficientnet_v2s": 384,
              "resnet50_places365": 224,
              "movinet_a0_base": 172,
              "movinet_a1_base": 172,
              "movinet_a2_base": 224,
              "movinet_a3_base": 256,
              "movinet_a4_base": 290,
              "movinet_a5_base": 320,
              "convnext_tiny_1k_224_fe": 224,
              "convnext_small_1k_224_fe": 224,
              "convnext_base_1k_224_fe": 224,
              "convnext_base_1k_384_fe": 384,
              "convnext_large_1k_224_fe": 224,
              "convnext_large_1k_384_fe": 384,
              "convnext_base_21k_1k_224_fe": 224,
              "convnext_base_21k_1k_384_fe": 384,
              "convnext_large_21k_1k_224_fe": 224,
              "convnext_large_21k_1k_384_fe": 384}

PREPROCESS_FUNCS = {"inception_v3": preprocess_input_inception_v3,
                    "mobilenet_v2": preprocess_input_mobilenet_v2,
                    "mobilenet_v3_large": preprocess_input_mobilenet_v3,
                    "xception": preprocess_input_xception,
                    "efficientnet_v2b3": preprocess_input_efficientnetV2,
                    "efficientnet_v2s": preprocess_input_efficientnetV2,
                    "resnet50_places365": preprocess_input_resnet50_places365,
                    "movinet_a0_base": preprocess_input_movinet,
                    "movinet_a1_base": preprocess_input_movinet,
                    "movinet_a2_base": preprocess_input_movinet,
                    "movinet_a3_base": preprocess_input_movinet,
                    "movinet_a4_base": preprocess_input_movinet,
                    "movinet_a5_base": preprocess_input_movinet,
                    "convnext_tiny_1k_224_fe": preprocess_input_convnext,
                    "convnext_small_1k_224_fe": preprocess_input_convnext,
                    "convnext_base_1k_224_fe": preprocess_input_convnext,
                    "convnext_base_1k_384_fe": preprocess_input_convnext,
                    "convnext_large_1k_224_fe": preprocess_input_convnext,
                    "convnext_large_1k_384_fe": preprocess_input_convnext,
                    "convnext_base_21k_1k_224_fe": preprocess_input_convnext,
                    "convnext_base_21k_1k_384_fe": preprocess_input_convnext,
                    "convnext_large_21k_1k_224_fe": preprocess_input_convnext,
                    "convnext_large_21k_1k_384_fe": preprocess_input_convnext}

BACKBONE_MODELS = {"inception_v3": InceptionV3,
                   "mobilenet_v2": MobileNetV2,
                   "mobilenet_v3_large": MobileNetV3Large,
                   "xception": Xception,
                   "efficientnet_v2b3": EfficientNetV2B3,
                   "efficientnet_v2s": EfficientNetV2S,
                   "resnet50_places365": ResNet50_Places365,
                   "movinet_a0_base": partial(Movinet, model_id="a0"),
                   "movinet_a1_base": partial(Movinet, model_id="a1"),
                   "movinet_a2_base": partial(Movinet, model_id="a2"),
                   "movinet_a3_base": partial(Movinet, model_id="a3"),
                   "movinet_a4_base": partial(Movinet, model_id="a4"),
                   "movinet_a5_base": partial(Movinet, model_id="a5"),
                   "convnext_tiny_1k_224_fe": partial(ConvNext, backbone_model="convnext_tiny_1k_224_fe"),
                   "convnext_small_1k_224_fe": partial(ConvNext, backbone_model="convnext_small_1k_224_fe"),
                   "convnext_base_1k_224_fe": partial(ConvNext, backbone_model="convnext_base_1k_224_fe"),
                   "convnext_base_1k_384_fe": partial(ConvNext, backbone_model="convnext_base_1k_384_fe"),
                   "convnext_large_1k_224_fe": partial(ConvNext, backbone_model="convnext_large_1k_224_fe"),
                   "convnext_large_1k_384_fe": partial(ConvNext, backbone_model="convnext_large_1k_384_fe"),
                   "convnext_base_21k_1k_224_fe": partial(ConvNext, backbone_model="convnext_base_21k_1k_224_fe"),
                   "convnext_base_21k_1k_384_fe": partial(ConvNext, backbone_model="convnext_base_21k_1k_384_fe"),
                   "convnext_large_21k_1k_224_fe": partial(ConvNext, backbone_model="convnext_large_21k_1k_224_fe"),
                   "convnext_large_21k_1k_384_fe": partial(ConvNext, backbone_model="convnext_large_21k_1k_384_fe")}
