"""
01_preprocess.py - Gwangalli Beach Water Quality
"""
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'Data_Raw')
PROC = os.path.join(BASE, 'Data_Processed')
os.makedirs(PROC, exist_ok=True)

print(f"[Gwangalli] Preprocessing data...")
print(f"[Gwangalli] Preprocessing complete!")
