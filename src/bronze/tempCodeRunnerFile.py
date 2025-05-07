from dotenv import load_dotenv
import os
import requests

load_dotenv()

print(os.getenv('OPENWEATHERMAP_API_KEY'))