import pandas as pd

df = pd.read_csv('baseball-reference data\hof-pitching.csv')
smallest_year = df['From'].min()
largest_year = df['From'].max()
print(f'Smallest: {smallest_year}')
print(f'Largest: {largest_year}')