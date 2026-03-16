import requests
import json
import random
import os
from pathlib import Path

OUT_DIR = Path("./random_words")
NUM_FILES = 1000
MAX_CHAR = 10000

data = requests.get("https://random-word-api.herokuapp.com/all")
if not data.ok:
    print("Failed to get data")

loaded_data: list[str] = data.json()


num_files = 0 
while num_files <= NUM_FILES:

    char_count: int = 0
    file_out: list[str] = []
    while char_count < MAX_CHAR:
        rand_num: int = int(len(loaded_data) * random.random())
        rand_word: str = loaded_data[rand_num]
        
        new_line: float = random.random()
        if new_line <= 0.05:
            rand_word = rand_word + "\n"

        file_out.append(rand_word)
        char_count += len(rand_word)
        print(f"\rFile size: {char_count}", end="")
    
    print(f"\nCreating file {num_files}.txt")

    out_file = OUT_DIR / f"{num_files}.txt"
    with open(out_file, 'w+') as file:
        file.write(' '.join(file_out))

    num_files += 1
    
