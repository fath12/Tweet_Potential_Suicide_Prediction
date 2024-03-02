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

logging.basicConfig(level=logging.INFO)

try:
    load_dotenv()

    model_file_path = os.getenv("MODEL_FILE_PATH")
    if model_file_path is None:
        raise EnvironmentError("MODEL_FILE_PATH not found in environment variables.")
except Exception as e:
    logging.error(f"Error loading environment variables: {e}")
    model_file_path = None

model = load_keras_model(model_file_path)

if model is None:
    logging.warning("Keras model is None after loading.")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    global model
    try:
        logging.info(f"Model status after startup: {model}")
        if model is None:
            logging.warning("Keras model is None after startup.")
    except Exception as e:
        logging.error(f"Error during startup: {e}")

create_db_and_tables()

def make_prediction(model, tweet):
    try:
        features = [[tweet]]

        prediction = model.predict(features)

        return prediction[0][0]
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
