import requests

api_key = "OASYJ7NakveKTgn4YHArbYp0zkZMiZuv"
symbol = "AAPL"
url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={api_key}"

response = requests.get(url)
print(response.json())
