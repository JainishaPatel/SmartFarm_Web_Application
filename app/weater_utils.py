import requests

def get_user_location():
    response = requests.get("http://ip-api.com/json/")
    data = response.json()
    return data.get("regionName", ""), data.get("lat"), data.get("lon")

def get_weather(lat, lon, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "rainfall": data.get("rain", {}).get("1h", 0.0)
    }