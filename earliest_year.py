import pandas as pd

df = pd.read_csv('baseball-reference data\hof-pitching.csv')
smallest_year = df['Inducted'].min()
largest_year = df['Inducted'].max()
print(f'Smallest: {smallest_year}')
print(f'Largest: {largest_year}')