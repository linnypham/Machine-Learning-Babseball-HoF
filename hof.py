from pybaseball import playerid_lookup, statcast_batter, statcast_pitcher
import datetime
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', message='Columns.*have mixed types')
# Load models
batting_model = joblib.load('models/batting_model_lr.pkl')
pitching_model = joblib.load('models/pitching_model_lr.pkl')

def is_pitcher_or_batter_all_time(first, last):
    # Remove leading/trailing spaces from player names
    first = first.strip()
    last = last.strip()
    try:
        player_info = playerid_lookup(last, first)
        if player_info.empty:
            return None, "Player not found"
        player_id = int(player_info.iloc[0]['key_mlbam'])
        start_date = '2015-01-01'
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        batter_data = statcast_batter(start_date, end_date, player_id)
        pitcher_data = statcast_pitcher(start_date, end_date, player_id)
        has_batting = not batter_data.empty
        has_pitching = not pitcher_data.empty
        if has_pitching and not has_batting:
            return 'pitcher', player_id
        elif has_batting and not has_pitching:
            return 'batter', player_id
        elif has_batting and has_pitching:
            return 'both', player_id
        else:
            return 'none', player_id
    except Exception as e:
        return None, str(e)

def calculate_batter_features(batter_data):
    if batter_data.empty:
        return None

    # Calculate true plate appearances
    pa = batter_data[['game_pk', 'inning', 'at_bat_number']].drop_duplicates().shape[0]

    # Hit detection with expanded types
    hit_types = ['single', 'double', 'triple', 'home_run',
                'fielders_choice', 'field_error', 'grounded_into_double_play']
    hits = batter_data['events'].isin(hit_types).sum()

    # Advanced metrics
    bb = batter_data['events'].isin(['walk', 'intent_walk']).sum()
    hbp = batter_data['events'].isin(['hit_by_pitch']).sum()
    total_bases = (batter_data['events'] == 'single').sum() * 1 + \
                  (batter_data['events'] == 'double').sum() * 2 + \
                  (batter_data['events'] == 'triple').sum() * 3 + \
                  (batter_data['events'] == 'home_run').sum() * 4

    # Handle RBI column naming variations
    rbi_column = next((col for col in ['rbi', 'rbis', 'RBI'] if col in batter_data.columns), None)
    rbi = batter_data[rbi_column].sum() if rbi_column else 0

    # Calculate rates
    avg = hits / pa if pa > 0 else 0
    obp = (hits + bb + hbp) / pa if pa > 0 else 0
    slg = total_bases / pa if pa > 0 else 0
    ops = obp + slg

    # Additional features from logistic regression model
    bb_rate = bb / pa if pa > 0 else 0

    # Barrel rate: check columns exist
    if 'launch_speed' in batter_data.columns and 'launch_angle' in batter_data.columns:
        barrel_rate = ((batter_data['launch_speed'] >= 98) &
                       (batter_data['launch_angle'].between(26, 30))).mean()
    else:
        barrel_rate = 0

    return [avg, obp, slg, ops, rbi, bb_rate, barrel_rate]

def calculate_pitcher_features(pitcher_data):
    if pitcher_data.empty:
        return None
    try:
        outs_recorded = pitcher_data['outs_when_up'].sum()
        innings_pitched = outs_recorded / 3

        # Handle multiple column name possibilities
        era_col = next((col for col in ['earned_run', 'ER', 'earned_runs', 'ERA']
                        if col in pitcher_data.columns), None)
        earned_runs = pitcher_data[era_col].sum() if era_col else 0
        era = (earned_runs * 9) / innings_pitched if innings_pitched > 0 else 0

        hits = pitcher_data['events'].isin(['single', 'double', 'triple', 'home_run']).sum()
        walks = pitcher_data['events'].isin(['walk', 'intent_walk']).sum()
        strikeouts = pitcher_data['events'].isin(['strikeout', 'strikeout_double_play']).sum()
        home_runs = pitcher_data['events'].isin(['home_run']).sum()

        whip = (walks + hits) / innings_pitched if innings_pitched > 0 else 0
        k9 = (strikeouts * 9) / innings_pitched if innings_pitched > 0 else 0
        fip = ((13 * home_runs) + (3 * walks) - (2 * strikeouts)) / innings_pitched + 3.1 if innings_pitched > 0 else 0
        kbb = strikeouts / walks if walks > 0 else 0

        return [era, whip, k9, fip, kbb]
    except KeyError as e:
        print(f"Missing column: {str(e)}")
        return None

def predict_player_hof(first, last):
    # Remove leading/trailing spaces from player names
    first = first.strip()
    last = last.strip()
    role, pid = is_pitcher_or_batter_all_time(first, last)
    if role is None:
        print(f"Error: {pid}")
        return

    if role in ['batter', 'both']:
        batter_data = statcast_batter('2015-01-01', datetime.datetime.now().strftime('%Y-%m-%d'), pid)
        if not batter_data.empty:
            batter_features = calculate_batter_features(batter_data)
            if batter_features:
                # Select first 5 features to match model training columns
                selected_features = batter_features[:5]  # ['AVG','OBP','SLG','OPS','RBI']
                features_df = pd.DataFrame([selected_features],
                                          columns=['AVG', 'OBP', 'SLG', 'OPS', 'RBI'])
                batter_pred = batting_model.predict(features_df)
                batter_proba = batting_model.predict_proba(features_df)[:, 1][0]
                print(f"\nBatter Prediction: {'HOF' if batter_pred[0] else 'Not HOF'}")
                print(f"Confidence: {batter_proba:.2%}")
            else:
                print("Insufficient batter data for feature calculation.")
        else:
            print("No recent batter data found.")

    if role in ['pitcher', 'both']:
        pitcher_data = statcast_pitcher('2015-01-01', datetime.datetime.now().strftime('%Y-%m-%d'), pid)
        if not pitcher_data.empty:
            pitcher_features = calculate_pitcher_features(pitcher_data)
            if pitcher_features:
                features_df = pd.DataFrame([pitcher_features],
                                          columns=['ERA', 'WHIP', 'K/9', 'FIP', 'K/BB'])
                pitcher_pred = pitching_model.predict(features_df)
                pitcher_proba = pitching_model.predict_proba(features_df)[:, 1][0]
                print(f"\nPitcher Prediction: {'HOF' if pitcher_pred[0] else 'Not HOF'}")
                print(f"Confidence: {pitcher_proba:.2%}")
            else:
                print("Insufficient pitcher data for feature calculation.")
        else:
            print("No recent pitcher data found.")

