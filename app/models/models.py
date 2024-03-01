from sqlalchemy import Column, Integer, Text, Float
from app.base import Base
from pydantic import BaseModel

class TweetData(Base):
    __tablename__ = "tweetdata"

    id = Column(Integer, primary_key=True, index=True)
    tweet = Column(Text(length=255), index=True)  # Adjust the length as needed


class ResultData(Base):
    __tablename__ = "resultdata"

    id = Column(Integer, primary_key=True, index=True)
    tweet = Column(Text(length=255), index=True)  # Adjust the length as needed
    prediction = Column(Float, nullable=True)

class TweetData_Pydantic(BaseModel):
    id: int
    tweet: str

    class Config:
        orm_mode = True

class ResultData_Pydantic(BaseModel):
    id: int
    tweet: str
    prediction: float

    class Config:
        orm_mode = True
