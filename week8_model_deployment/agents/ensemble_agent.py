import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from agents.agent import Agent
import statistics

from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.random_forest_agent import RandomForestAgent
from agents.lightgbm_agent import LightGBMAgent
from sklearn.exceptions import InconsistentVersionWarning

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class EnsembleAgent(Agent):

    name = "Ensemble Agent"
    color = Agent.YELLOW
    
    def __init__(self, collection):
        """
        Create an instance of Ensemble, by creating each of the models
        And loading the weights of the Ensemble
        """
        self.log("Initializing Ensemble Agent")
        self.specialist = SpecialistAgent()
        self.frontier_1 = FrontierAgent(collection, provider='ollama', local_model_name='deepseek-r1:8b')
        self.frontier_2 = FrontierAgent(collection, provider='ollama', local_model_name='qwen3:8b')
        self.random_forest = RandomForestAgent()
        self.lightgbm = LightGBMAgent()
        self.model = joblib.load(f"{MODEL_DIR}/ensemble_model_5x.pkl")
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        """
        Run this ensemble model
        Ask each of the models to price the product
        Then use the Linear Regression model to return the weighted price
        :param description: the description of a product
        :return: an estimate of its price
        """
        self.log("Running Ensemble Agent - collaborating with specialist, frontier and random forest agents")
        specialist = self.specialist.price(description)
        frontier_1 = self.frontier_1.price(description)
        frontier_2 = self.frontier_2.price(description)
        random_forest = self.random_forest.price(description)
        lightgbm = self.lightgbm.price(description)

        values = [specialist, frontier_1, frontier_2, random_forest, lightgbm]

        X = pd.DataFrame({
            'Specialist': [specialist],
            'Frontier 1': [frontier_1],
            'Frontier 2': [frontier_2],
            'RandomForest': [random_forest],
            'LightGBM': [lightgbm],
            'Min': [min(values)],
            'Max': [max(values)],
        })
        y_lr = max(0, self.model.predict(X)[0])
        y_mean = max(0, statistics.mean(values))
        y_median = max(0, statistics.median(values))

        self.log(f"Ensemble Agent complete - returning ${y_lr:.2f} (lr) - ${y_mean:.2f} (mean) - ${y_median:.2f} (median).")
        return y_lr, y_mean, y_median