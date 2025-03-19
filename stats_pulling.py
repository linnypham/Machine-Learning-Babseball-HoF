from pybaseball import batting_stats,pitching_stats
from datetime import date
#pull data from 1936 to current year
data_batting = batting_stats(1936,date.today().year)
data_pitching = pitching_stats(1936,date.today().year)
data_batting.to_csv('baseball-reference data/all_batting.csv',index=False)
data_pitching.to_csv('baseball-reference data/all_pitching.csv',index=False)
print('Done!!!')