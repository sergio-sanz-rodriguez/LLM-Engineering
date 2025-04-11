import openai
import os
from dotenv import load_dotenv
from openai import OpenAI

# Set up your API key
# It's best practice to use environment variables for API keys
#openai.api_key = os.getenv("OPENAI_API_KEY")
#set OPENAI_API_KEY=your-api-key-here

def generate_text(user_prompt, model="gpt-4o", max_tokens=1000):

    system_prompt = "You are coding assistant that explains what a Python code does and why."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    if "gpt" in model:
        try:
            # Set API key only if using OpenAI GPT
            load_dotenv(override=True)            
            api_key = os.getenv("OPENAI_API_KEY")
            openai = OpenAI()
            
            if not openai.api_key:
                return "OPENAI_API_KEY not found in environment variables."
        
            # Create a chat completion
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                #max_tokens=max_tokens
            )
            
            # Extract and return the generated text
            return response.choices[0].message.content
        
        except Exception as e:
            return f"An error occurred: {str(e)}"
            
    elif "llama" in model:

        try:
            # Import ollama
            import ollama
            
            # Create chat
            response = ollama.chat(
                model=model,
                messages=messages
            )            

            # Extract and return the generated text
            return response['message']['content'].strip()
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
            
    else:
        return f"Unsupported model: {model}"
                
# Interactive usage
if __name__ == "__main__":
    print("Welcome to the AI Assistant!")
    model = input("Enter model (e.g., gpt-4o, gpt-3.5-turbo, llama3.2, etc.): ").strip()

    print("Type your prompt below (or type 'exit' to quit):")
    while True:
        user_prompt = input("\nYour prompt: ")

        if user_prompt.lower() in ["exit", "quit", "q"]:
            print("Exiting program. Goodbye!")
            break

        print("\nGenerating response...\n")
        result = generate_text(user_prompt, model=model)
        print("Generated Response:")
        print(result)
        print("\n" + "-" * 50)