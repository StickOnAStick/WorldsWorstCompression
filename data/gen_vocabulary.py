import requests
import json


data = requests.get(url="https://random-word-api.herokuapp.com/all")
if not data.ok:
    print("Could not load API data")
    exit(1)

loaded_data = data.json()

with open('./vocab/english_words.txt', "+w") as file:
    if not file:
        print("Could not open file")
        exit(1)
    file.writelines('\n'.join(loaded_data))


file.close()