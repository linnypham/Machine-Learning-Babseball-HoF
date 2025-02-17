from pybaseball import playerid_lookup
import pandas as pd

def get_hof_players():
    # Get all players
    all_players = playerid_lookup(' ', fuzzy=True)
    
    # Filter for Hall of Fame players
    hof_players = all_players[all_players['hof'] == 'Y']
    
    # Sort by induction year and name
    hof_players = hof_players.sort_values(['hof_year', 'name_first', 'name_last'])
    
    # Select relevant columns, including player_id
    columns = ['key_mlbam', 'name_first', 'name_last', 'hof_year', 'mlb_played_first', 'mlb_played_last']
    hof_players = hof_players[columns]
    
    # Rename columns for clarity
    hof_players = hof_players.rename(columns={
        'key_mlbam': 'player_id',
        'name_first': 'first_name',
        'name_last': 'last_name',
        'hof_year': 'induction_year',
        'mlb_played_first': 'career_start',
        'mlb_played_last': 'career_end'
    })
    
    return hof_players

# Get the Hall of Fame players
hof_players = get_hof_players()

# Display the first few rows
print(hof_players.head())

# Save to CSV
hof_players.to_csv('data\baseball_hall_of_fame_players.csv', index=False)
print("CSV file 'hall_of_fame_players.csv' has been created successfully.")
