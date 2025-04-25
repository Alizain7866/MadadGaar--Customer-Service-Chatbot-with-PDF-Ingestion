from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
import os
from dotenv import load_dotenv

load_dotenv() 

# Set these as environment variables or replace with your actual values
endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
key = os.getenv("AZURE_CONTENT_SAFETY_KEY")

if key is None:
    raise ValueError("AZURE_CONTENT_SAFETY_KEY is not set in environment variables.")
if endpoint is None:
    raise ValueError("AZURE_CONTENT_SAFETY_ENDPOINT is not set in environment variables.")

# If the key and endpoint are valid, create the client
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential

client = ContentSafetyClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def is_safe(text, threshold=3):
    """
    Analyze the text to ensure it is safe based on the threshold.
    Returns True if safe, False if not.
    """
    options = AnalyzeTextOptions(text=text)
    result = client.analyze_text(options)
    
    for category in result.categories_analysis:
        if category.severity > threshold:
            return False  # Unsafe content
    return True  # Safe content
