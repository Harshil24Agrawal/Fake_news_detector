import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("Loading original datasets...")
fake_df = pd.read_csv('layer1_pattern/Fake.csv')
true_df = pd.read_csv('layer1_pattern/True.csv')

# Standardize labels: 1 for FAKE, 0 for REAL (common across layers)
fake_df['label'] = 1
true_df['label'] = 0

df = pd.concat([fake_df, true_df], ignore_index=True)

# Basic cleaning to ensure no empty rows
df['combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df = df[df['combined'].str.strip() != '']

print(f"Total valid articles: {len(df)}")

# Split 70% Train, 15% Validation, 15% Test
print("Splitting data into global train, val, and test sets...")
train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])

train.to_csv('global_train.csv', index=False)
val.to_csv('global_val.csv', index=False)
test.to_csv('global_test.csv', index=False)

print("Created global_train.csv, global_val.csv, and global_test.csv successfully!")
