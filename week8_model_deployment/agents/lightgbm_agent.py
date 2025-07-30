# imports

import os
import re
from typing import List
from sentence_transformers import SentenceTransformer
import joblib
from agents.agent import Agent
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


class LightGBMAgent(Agent):

    name = "LightGBM Agent"
    color = Agent.CYAN

    def __init__(self):
        """
        Initialize this object by loading in the saved model weights
        and the SentenceTransformer vector encoding model
        """
        self.log("LightGBM Agent is initializing")
        self.vectorizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.model = joblib.load(f"{MODEL_DIR}/lgb_model_2000_lr0.1.pkl")
        self.log("LightGBM Agent is ready")

    def price(self, description: str) -> float:
        """
        Use a LightGBM model to estimate the price of the described item
        :param description: the product to be estimated
        :return: the price as a float
        """        
        self.log("LightGBM Agent is starting a prediction")
        vector = self.vectorizer.encode([description])
        result = max(0, self.model.predict(vector)[0])
        self.log(f"LightGBM Agent completed - predicting ${result:.2f}")
        return result