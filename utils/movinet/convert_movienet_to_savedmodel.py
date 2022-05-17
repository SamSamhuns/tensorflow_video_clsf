from dotenv import load_dotenv
load_dotenv(".env")
import tensorflow as tf
import tensorflow_hub as hub
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

"""
Notebook:
    https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb

Clone the GitHub repository:
    git clone https://github.com/tensorflow/models.git
Add the top-level /models folder to the Python path.
    export PYTHONPATH=$PYTHONPATH:/path/to/models

kinetics-600
Model Name	    Top-1 Acc Top-5 Acc	Input Shape	     GFLOPs FPS     URL(prefix=https://tfhub.dev/tensorflow/movinet/)
MoViNet-A0-Base	72.28	  90.92	    50 x 172 x 172	 2.7    5 fps   a0/base/kinetics-600/classification/3
MoViNet-A1-Base	76.69	  93.40	    50 x 172 x 172	 6.0    5 fps   a1/base/kinetics-600/classification/3
MoViNet-A2-Base	78.62	  94.17	    50 x 224 x 224	 10     5 fps   a2/base/kinetics-600/classification/3
MoViNet-A3-Base	81.79	  95.67	    120 x 256 x 256	 57     12 fps  a3/base/kinetics-600/classification/3
MoViNet-A4-Base	83.48	  96.16	    80 x 290 x 290	 110    8 fps   a4/base/kinetics-600/classification/3
MoViNet-A5-Base	84.27	  96.39	    120 x 320 x 320	 280    12 fps  a5/base/kinetics-600/classification/3

"""


def movinet_model_test():
    hub_url = "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"

    encoder = hub.KerasLayer(hub_url, trainable=True)

    inputs = tf.keras.layers.Input(
        shape=[None, None, None, 3],
        dtype=tf.float32,
        name='image')

    # [batch_size, 600]
    outputs = encoder(dict(image=inputs))

    model = tf.keras.Model(inputs, outputs, name='movinet')
    model.summary()

    x = model.layers[-1].output
    x = tf.keras.layers.Dense(256)(x)
    predictions = tf.keras.layers.Dense(15, activation="softmax")(x)
    model2 = tf.keras.models.Model(inputs=model.input, outputs=predictions)

    model2.summary()

    example_input = tf.ones([1, 8, 224, 224, 3])
    example_output = model(example_input)

    print(example_output.shape)


def movinet_backbone_test():
    # tf.keras.backend.clear_session()
    model_id = 'a0'

    backbone = movinet.Movinet(model_id=model_id)
    model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
    model.build([1, 1, 1, 1, 3])
    checkpoint_dir = 'movinet_a0_base'
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    status.assert_existing_objects_matched()
    model.summary()

    batch_size = 8
    num_frames = 8
    resolution = 172
    num_classes = 23
    # frame_stride = 10
    # num_epochs = 3
    # initial_learning_rate = 0.01

    def build_classifier(backbone, num_classes, freeze_backbone=False):
        """Builds a classifier on top of a backbone model."""
        model = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=num_classes)
        model.build([batch_size, num_frames, resolution, resolution, 3])

        if freeze_backbone:
            for layer in model.layers[:-1]:
                layer.trainable = False
            model.layers[-1].trainable = True

        return model

    model = build_classifier(backbone, num_classes, freeze_backbone=True)
    model.summary()

    # train_steps = num_examples['train'] // batch_size
    # total_train_steps = train_steps * num_epochs
    # test_steps = num_examples['test'] // batch_size
    #
    # loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    # learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=total_train_steps)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)


movinet_model_test()
