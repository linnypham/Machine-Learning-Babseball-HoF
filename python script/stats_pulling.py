from pybaseball import batting_stats_bref,pitching_stats_bref
from datetime import date
#pull data from 1936 to current year
data_batting = batting_stats_bref(1936,date.today().year)
data_pitching = pitching_stats_bref(1936,date.today().year)
data_batting.to_csv('baseball-reference data/all_batting.csv',index=False)
data_pitching.to_csv('baseball-reference data/all_pitching.csv',index=False)
print('Done!!!')