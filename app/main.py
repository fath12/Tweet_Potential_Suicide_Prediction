import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.models import TweetData, ResultData, TweetData_Pydantic, ResultData_Pydantic
from app.database.database import engine, get_db, create_db_and_tables
from app.model_loader import load_keras_model
import uvicorn
import tensorflow as tf
from fastapi.staticfiles import StaticFiles

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configure logging
logging.basicConfig(level=logging.INFO)

try:
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve model file path from environment variables
    model_file_path = os.getenv("MODEL_FILE_PATH")
    if model_file_path is None:
        raise EnvironmentError("MODEL_FILE_PATH not found in environment variables.")
except Exception as e:
    logging.error(f"Error loading environment variables: {e}")
    model_file_path = None

# Load the Keras model
model = load_keras_model(model_file_path)

# Print a message if the model is not loaded
if model is None:
    logging.warning("Keras model is None after loading.")

# Initialize FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Event handler to load Keras model on startup
@app.on_event("startup")
async def startup_event():
    global model
    try:
        # No need to call load_keras_model here, as it's already loaded during the app startup
        logging.info(f"Model status after startup: {model}")
        if model is None:
            logging.warning("Keras model is None after startup.")
    except Exception as e:
        logging.error(f"Error during startup: {e}")

# Creates all the tables defined in models module
create_db_and_tables()

# Make prediction function
def make_prediction(model, tweet):
    try:
        # Make an input vector
        features = [[tweet]]

        # Predict
        prediction = model.predict(features)

        return prediction[0][0]
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

# Insert Prediction information
def insert_prediction(tweet, prediction, db):
    new_prediction = ResultData(
        tweet=tweet,
        prediction=prediction
    )

    with db as session:
        session.add(new_prediction)
        session.commit()
        session.refresh(new_prediction)

    return new_prediction 

# Endpoint for predicting
@app.post("/prediction/suicide", response_model=ResultData_Pydantic)
async def predict_suicide(request: TweetData_Pydantic, db: Session = Depends(get_db)):
    try:
        tweet = request.tweet
        prediction = make_prediction(model, tweet)
        db_record = insert_prediction(tweet=tweet, prediction=prediction, db=db)
        return ResultData_Pydantic(id=db_record.id, tweet=db_record.tweet, prediction=db_record.prediction)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
