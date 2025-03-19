from pybaseball import playerid_lookup
import pandas as pd

# Load dataset (modify the file path as needed)
df = pd.read_csv('baseball-reference data/all_pitching.csv')

# Split 'Name' column into 'first_name' and 'last_name'
df[['first_name', 'last_name']] = df['Name'].str.rsplit(' ', n=1, expand=True)

# Function to get playerID
def get_player_id(row):
    try:
        result = playerid_lookup(row['last_name'], row['first_name'])
        if not result.empty:
            return result.iloc[0]['key_bbref']  # Use Baseball Reference ID
    except:
        return None
    return None

# Apply function to create 'playerID' column
df['playerID'] = df.apply(get_player_id, axis=1)

# Save updated dataset
df.to_csv('baseball-reference data/all_pitching_updated.csv', index=False)

print("playerID column added successfully!")
