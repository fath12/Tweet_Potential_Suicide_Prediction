import os
import signal
import sys
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from sqlalchemy.orm import Session
from app.models.models import TweetData, ResultData
from app.database.database import get_db, engine
from app.main import make_prediction

load_dotenv()

model_file_path = os.getenv("MODEL_FILE_PATH")

model = None

def load_keras_model():
    global model
    if model_file_path:
        try:
            model = load_model(model_file_path)
            print("Keras model loaded successfully")
        except Exception as e:
            print(f"Error loading Keras model: {e}")
    else:
        print("Keras model not loaded. Model file path is not available.")

def auto_predict():
    db = next(get_db())

    try:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        new_data = db.query(TweetData).filter(TweetData.id.notin_(db.query(ResultData.id))).all()

        for data in new_data:
            prediction = make_prediction(model, data.tweet)

            result_data = ResultData(tweet=data.tweet, prediction=prediction)
            db.add(result_data)
            db.commit()

            break
    finally:
        db.close()

def signal_handler(sig, frame):
    print("Received interrupt signal. Exiting gracefully.")
    sys.exit(0)

if __name__ == "__main__":
    load_keras_model()
    signal.signal(signal.SIGINT, signal_handler)

    auto_predict()
