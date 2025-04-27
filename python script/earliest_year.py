import pandas as pd

df = pd.read_csv('baseball-reference data/all_batting_updated.csv')
smallest_year = df['Season'].min()
largest_year = df['Season'].max()
print(f'Smallest: {smallest_year}')
print(f'Largest: {largest_year}')