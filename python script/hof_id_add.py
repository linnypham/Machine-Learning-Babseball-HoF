import pandas as pd
import csv
from pybaseball import playerid_lookup

# Read the CSV file and update rows
with open('data/hof_players.csv', mode='r', newline='') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['mlb_id']  # Add 'mlb_id' to the header
    rows = list(reader)  # Read all rows into a list

    # Update each row with the MLB ID
    for row in rows:
        player = row['Player'].split()
        data = playerid_lookup(first=player[0], last=player[1])
        if not data.empty:
            row['mlb_id'] = data.iloc[0]['key_mlbam']

        # Write the updated data back to the CSV file
with open('data/hof_players.csv', mode='w', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()  # Write the header
    writer.writerows(rows)  # Write all rows

hof_player = pd.read_csv('data/hof_players.csv')
print(hof_player)