import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_gemini_api():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables")
        return
    
    print(f"API Key loaded: {'Present' if api_key else 'Missing'}")
    print(f"API Key length: {len(api_key)}")
    
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        print("Successfully configured Gemini API")
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-pro')
        print("Successfully initialized Gemini model")
        
        # Test the model with a simple prompt
        response = model.generate_content("Hello, this is a test message.")
        print("Successfully generated content")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_gemini_api() 