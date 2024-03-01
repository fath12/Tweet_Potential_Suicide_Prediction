import os
import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

def load_keras_model(model_file_path):
    try:
        # Check if the SavedModel directory exists
        model_directory = os.path.join(model_file_path)
        if not os.path.exists(model_directory):
            raise FileNotFoundError(f"SavedModel directory not found at: {model_directory}")

        # Load the Keras model using tensorflow.keras.models.load_model
        model = tf.keras.models.load_model(model_directory)

        logging.info("Keras model loaded successfully")
        return model
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading Keras model: {e}")
        return None
