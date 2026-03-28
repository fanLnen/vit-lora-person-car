
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "logs")

print(BASE_DIR,OUTPUT_DIR,MODEL_OUTPUT_DIR,LOG_OUTPUT_DIR)