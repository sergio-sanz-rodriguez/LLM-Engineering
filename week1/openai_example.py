import openai
import os

# Set up your API key
# It's best practice to use environment variables for API keys
#openai.api_key = os.getenv("OPENAI_API_KEY")
#set OPENAI_API_KEY=your-api-key-here

def generate_text(prompt, model="gpt-4o", max_tokens=500):
    """
    Generate text using OpenAI's API
    
    Args:
        prompt: The text prompt to send to the API
        model: The model to use (default: gpt-4o)
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        The generated text response
    """
    try:
        # Create a chat completion
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        
        # Extract and return the generated text
        return response.choices[0].message.content
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Interactive usage
if __name__ == "__main__":
    print("Welcome to the OpenAI API example!")
    print("Type your prompt below (or type 'exit' to quit):")
    
    while True:
        user_prompt = input("\nYour prompt: ")
        
        # Check if user wants to exit
        if user_prompt.lower() in ["exit", "quit", "q"]:
            print("Exiting program. Goodbye!")
            break
        
        # Generate and display response
        print("\nGenerating response...\n")
        result = generate_text(user_prompt)
        print("Generated Response:")
        print(result)
        print("\n" + "-"*50)