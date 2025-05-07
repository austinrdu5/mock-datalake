from dotenv import load_dotenv
import os
import requests

load_dotenv()

print(os.getenv('OPENWEATHERMAP_API_KEY'))

lat = 51.5074
lon = -0.1278
dt = 1715025600
api_key = os.getenv('OPENWEATHERMAP_API_KEY')

response = requests.get("https://api.openweathermap.org/data/2.5/weather?id=524901&appid={api_key}".format(
    api_key=api_key
))
print(response.json())

response = requests.get("https://history.openweathermap.org/data/3.0/history/timemachine?lat={lat}&lon={lon}&dt={dt}&appid={api_key}".format(
    lat=lat,
    lon=lon,
    dt=dt,
    api_key=api_key
))

print(response.json())