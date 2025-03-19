import pandas as pd

# Load dataset
hof_batting = pd.read_csv('baseball-reference data/hof-pitching.csv')

# Rename the column
hof_batting.rename(columns={"Name-additional": "playerID"}, inplace=True)

# Save the updated dataset
hof_batting.to_csv('baseball-reference data/hof_pitching_updated.csv', index=False)

print("Column 'Name-additional' successfully renamed to 'playerID'!")
