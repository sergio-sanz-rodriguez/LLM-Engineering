# imports

import os
import re
import math
import json
import ollama
import subprocess
import requests
import time
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from items import Item
from testing import Tester
from agents.agent import Agent
from huggingface_hub import InferenceClient

class FrontierAgent(Agent):

    name = "Frontier Agent"
    color = Agent.BLUE

    MODEL = "gpt-4o-mini"
    
    def __init__(self, collection, provider='openai', local_model_name=None, n_results=5):
        """
        Set up this instance by connecting to different AI models remotely or locally, to the Chroma Datastore,
        And setting up the vector encoding model
        """
        self.client_type = provider
        if not isinstance(n_results, int):
            raise ValueError("`n_results` must be an integer representing the number of documents to retrieve in the RAG process.")
        self.n_results_rag = n_results
        if provider == 'deepseek': 
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            self.log("Initializing Frontier Agent")
            self.client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
            self.MODEL = "deepseek-chat"
            self.log("Frontier Agent is set up with DeepSeek")
        elif provider == 'openai': #Paid
            self.client = OpenAI()
            self.MODEL = "gpt-4o-mini"
            self.log("Frontier Agent is setting up with OpenAI")
        elif provider == 'llama': #Paid
            groq_api_key=os.getenv("GROQ_API_KEY")
            self.client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
            self.MODEL = "llama3-70b-8192"  # or "llama3-8b-8192" if you prefer
            self.log("Frontier Agent is setting up with LLama via Groq")
        elif provider == 'mistral': #Paid
            hf_api_token = os.getenv("HF_TOKEN")                        
            self.client = InferenceClient(provider="novita",api_key=hf_api_token)
            self.MODEL = "mistralai/Mistral-7B-Instruct-v0.3" #"mistralai/Mistral-7B-Instruct"
            self.log("Frontier Agent is setting up with Mistral via Hugging Face")
        elif provider == 'ollama': #Free, running locally
            # Activate Ollama
            self.activate_ollama()
            #Check out first the models available locally with "ollama list"
            if local_model_name is None:
                raise ValueError("You must specify local_model_name when using provider='ollama'")
            self.MODEL = local_model_name
            self.client = OpenAI(base_url="http://localhost:11434/v1")
            self.log(f"Frontier Agent is set up with Ollama using model '{local_model_name}'")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        self.collection = collection
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.log("Frontier Agent is ready")

    def is_ollama_running(self, url="http://localhost:11434"):
        try:
            response = requests.get(url)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def activate_ollama(self):
        if not self.is_ollama_running():
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Wait for server to start
            for _ in range(10):
                if self.is_ollama_running():
                    self.log("Ollama is now running.")
                    return
                time.sleep(1)
            raise RuntimeError("Ollama failed to start.")
        else:
            self.log("Ollama is already running.")
            
    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar products to the one being estimated
        :param prices: prices of the similar products
        :return: text to insert in the prompt that provides context
        """
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to OpenAI
        With the system and user prompt
        :param description: a description of the product
        :param similars: similar products to this one
        :param prices: prices of similar products
        :return: the list of messages in the format expected by OpenAI
        """
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + description
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]

    def find_similars(self, description: str):
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """
        self.log("Frontier Agent is performing a RAG search of the Chroma datastore to find similar products")
        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=self.n_results_rag)
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s) -> float:
        """
        A utility that plucks a floating point number out of a string
        """
        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        """
        Make a call to OpenAI or DeepSeek to estimate the price of the described product,
        by looking up 5 similar products and including them in the prompt to give context
        :param description: a description of the product
        :return: an estimate of the price
        """
        similars, prices = self.find_similars(description)
        self.log(f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products")
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL, 
                messages=self.messages_for(description, similars, prices),
                seed=42,
                max_tokens=5
            )
        except Exception as e:
            self.log(f"Error calling model: {e}")
            raise
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
        