from os import listdir
from tqdm import tqdm
from globals import *



def format_data(historical_data=True, current_season_data=False):
    """Formats the raw data from ncaahoopR into a single CSV file with one row per game.
       Each row contains data for both teams in the game."""
    rows = [",".join(["season", "date", 
            "team1id", "team1home", "team1pts", "team1fgm", "team1fga", "team13ptm", "team13pta", "team1ftm", "team1fta", "team1oreb", "team1dreb", "team1ast", "team1stl", "team1blk", "team1to", "team1pf",
            "team2id", "team2home", "team2pts", "team2fgm", "team2fga", "team23ptm", "team23pta", "team2ftm", "team2fta", "team2oreb", "team2dreb", "team2ast", "team2stl", "team2blk", "team2to", "team2pf"
        ]) + "\n"]

    games = {} # Dictionary from gameID to list of datarows

    # count = 0

    # Loop over every file in the directory ncaahoopR_data
    years = listdir("ncaahoopR_data")

    for year in tqdm(years):
        # Don't do some seasons, depending on flags
        if not current_season_data and year == currentSeason:
            continue
        if not historical_data and year != currentSeason:
            continue
        # Now, iterate through each school folder in year/box_scores
        try:
            schools = listdir(f"ncaahoopR_data/{year}/box_scores")
        except FileNotFoundError:
            continue # Probably not a year
        for school in tqdm(schools, desc=f"Processing year {year}"):
            # Iterate through each game file in year/box_scores/school
            gameIds = listdir(f"ncaahoopR_data/{year}/box_scores/{school}")
            for gameId in gameIds:
                with open(f"ncaahoopR_data/{year}/box_scores/{school}/{gameId}", "r") as f:
                    data = f.readlines()
                    
                    headers = data[0].strip().split(",")
                    teamData = data[-1].split(",") # Last row
                    # If the last row's player name isn't "TEAM", then that file doesn't have total stats. 
                    # Instead, we have to sum the numerical values for all the players, 
                    # and can use home/away values as well as dates, season, etc. of just one player
                    # (since those should be the same for all players in the same game)
                    if teamData[headers.index("player")] != "TEAM":
                        teamData = data[1].split(",") # First row after header
                        teamData[headers.index("player")] = "TEAM"
                        for line in data[2:]:
                            playerData = line.split(",")
                            for i in range(len(playerData)):
                                if playerData[i].isdigit():
                                    teamData[i] = str(int(teamData[i]) + int(playerData[i]))
                    
                    if gameId not in games: # First half of the game
                        season = year
                        date = teamData[headers.index("date")]
                        team2id = teamData[headers.index("opponent")]
                        try:
                            team1home = True if teamData[headers.index("home")] == "TRUE" else False
                        except ValueError:
                            try:
                                team1home = True if teamData[headers.index("location")] == "H" else False
                            except ValueError:
                                team1home = False
                        team1pts = teamData[headers.index("PTS")]
                        team1fgm = teamData[headers.index("FGM")]
                        team1fga = teamData[headers.index("FGA")]
                        team13ptm = teamData[headers.index("3PTM")]
                        team13pta = teamData[headers.index("3PTA")]
                        team1ftm = teamData[headers.index("FTM")]
                        team1fta = teamData[headers.index("FTA")]
                        team1oreb = teamData[headers.index("OREB")]
                        team1dreb = teamData[headers.index("DREB")]
                        team1ast = teamData[headers.index("AST")]
                        team1stl = teamData[headers.index("STL")]
                        team1blk = teamData[headers.index("BLK")]
                        team1to = teamData[headers.index("TO")]
                        team1pf = teamData[headers.index("PF")]
                        
                        team1Data = [season, date, team2id, team1home, team1pts, team1fgm, team1fga, team13ptm, team13pta, team1ftm, team1fta, team1oreb, team1dreb, team1ast, team1stl, team1blk, team1to, team1pf]
                        
                        # Check if any of these are NA, and if so, skip this game
                        if "NA" in team1Data:
                            # print(team1Data.index("NA"), gameId, season)
                            continue
                        
                        games[gameId] = team1Data
                        # print(f"Added first half of game {gameId}. Data: {games[gameId]}")
                    else:
                        team1id = teamData[headers.index("opponent")]
                        try:
                            team2home = True if teamData[headers.index("home")] == "TRUE" else False
                        except ValueError:
                            try:
                                team2home = True if teamData[headers.index("location")] == "H" else False
                            except ValueError:
                                team2home = False
                        team2pts = teamData[headers.index("PTS")]
                        team2fgm = teamData[headers.index("FGM")]
                        team2fga = teamData[headers.index("FGA")]
                        team23ptm = teamData[headers.index("3PTM")]
                        team23pta = teamData[headers.index("3PTA")]
                        team2ftm = teamData[headers.index("FTM")]
                        team2fta = teamData[headers.index("FTA")]
                        team2oreb = teamData[headers.index("OREB")]
                        team2dreb = teamData[headers.index("DREB")]
                        team2ast = teamData[headers.index("AST")]
                        team2stl = teamData[headers.index("STL")]
                        team2blk = teamData[headers.index("BLK")]
                        team2to = teamData[headers.index("TO")]
                        team2pf = teamData[headers.index("PF")]
                        
                        # Check if any of these are NA, and if so, delete this game
                        team2Data = [team1id, team2home, team2pts, team2fgm, team2fga, team23ptm, team23pta, team2ftm, team2fta, team2oreb, team2dreb, team2ast, team2stl, team2blk, team2to, team2pf]
                        if "NA" in team2Data:
                            del games[gameId]
                            continue
                        # There are some bizzarre weird cases where both teams are "home." These are cursed games and other things are wrong with them as well. Delete them.
                        if games[gameId][3] == True and team2home == True:
                            del games[gameId]
                            continue
                        games[gameId].extend(team2Data)
                        
                        # Swap team1index and team2index
                        team2id = games[gameId][2]
                        games[gameId][2] = team1id
                        games[gameId][18] = team2id
                        # print(gameId, "completed. Data:", games[gameId])
                    # count += 1
                    # if count > 100:
                    #     print(games)
                    #     exit(0)

    for game in tqdm(games):
        # if len(games[game]) != len(rows):
        #     print(f"Game {game} has incomplete data, skipping.")
        #     continue
        line = ",".join([str(x) for x in games[game]]) + "\n"
        rows.append(line)
        
    # Open the data file corresponding to the flag in append mode
    if historical_data:
        dataFile = historicalDataFile
    else:
        dataFile = currentSeasonDataFile
    with open(dataFile, "a") as f:
        for line in rows:
            # One last check to make sure line is complete
            fields = line.strip().split(",")
            if len(fields) != 34:
                print(f"Found incomplete line: {line}")
                continue
            # There's this one line that had a Duke vs WF game where it says Duke scored 18 (they actually scored 101). That line is just bizzarre
            if fields[2] == "Duke" and fields[18] == "Wake Forest" and fields[4] == "18":
                print(f"Deleting bizzarre Duke vs Wake Forest line: {line}")
                continue
            f.write(line)
    f.close()
    
if __name__ == "__main__":
    format_data(historical_data=True, current_season_data=False)
    # format_data(historical_data=False, current_season_data=True)