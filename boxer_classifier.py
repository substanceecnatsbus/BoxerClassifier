import numpy as np
from PIL import Image
import tensorflow as tf

tf.autograph.set_verbosity(3)


class BoxerClassifier:
    def __init__(self, image_size, model_path) -> None:
        self.image_size = image_size
        self.__model = BoxerClassifier.create_model(image_size)
        self.__model.load_weights(model_path).expect_partial()
        self.__classes = ['boxer-blue', 'boxer-red', 'others']

    @property
    def model(self):
        return self.__model

    def predict(self, image_np):
        prediction = self.model(image_np).numpy()
        index = np.argmax(prediction)
        result = self.__classes[index]
        return result

    @staticmethod
    def load_dataset(dataset_path, image_size, batch_size=16, test_split=0.1, seed=123):
        if test_split >= 1:
            raise Exception("test_split must be less than 1")

        train_dataset = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            subset="training",
            validation_split=test_split,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            seed=seed,
            label_mode="categorical"
        )
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            subset="validation",
            validation_split=test_split,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            seed=seed,
            label_mode="categorical"
        )
        class_names = train_dataset.class_names

        return train_dataset, test_dataset, class_names

    @staticmethod
    def create_model(image_size):
        base_model = tf.keras.applications.resnet.ResNet152(
            input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet",
            input_tensor=None, pooling=None,
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(image_size, image_size, 3))
        x = tf.keras.applications.resnet.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(
            3, activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    @staticmethod
    def train(image_size, dataset_path, save_path="", number_of_epochs=100, learning_rate=3e-3, test_split=0.1):
        model = BoxerClassifier.create_model(image_size)

        train_dataset, test_dataset, _ = BoxerClassifier.load_dataset(
            dataset_path, image_size, test_split=test_split)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        history = model.fit(train_dataset,
                            epochs=number_of_epochs,
                            validation_data=test_dataset)

        if save_path != "":
            model.save_weights(save_path)
        return (model, history)
