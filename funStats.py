from globals import *
from tqdm import tqdm

def largest_point_differential():
    with open(historicalDataFile, "r") as f:
        data = f.readlines()
    max_diff = 0
    for line in tqdm(data[1:]): # Skip header
        fields = line.strip().split(",")
        team1pts = int(fields[4])
        team2pts = int(fields[20])
        diff = abs(team1pts - team2pts)
        if diff > max_diff:
            max_diff = diff
            max_fields = fields 
    print(f"The largest point differential achieved was {max_diff} on {max_fields[1]}. The final score was {max_fields[2]} {max_fields[4]} - {max_fields[18]} {max_fields[20]}.")

def most_points_scored():
    with open(historicalDataFile, "r") as f:
        data = f.readlines()
    max_points = 0
    for line in tqdm(data[1:]): # Skip header
        fields = line.strip().split(",")
        totalPoints = int(fields[4]) + int(fields[20])
        if totalPoints > max_points:
            max_points = totalPoints
            max_fields = fields        
    print(f"The most points scored in a single game was {max_points} on {max_fields[1]}. The final score was {max_fields[2]} {max_fields[4]} - {max_fields[18]} {max_fields[20]}.")

def fewest_team_fouls():
    with open(historicalDataFile, "r") as f:
        data = f.readlines()
    min_fouls = float('inf')
    for line in tqdm(data[1:]): # Skip header
        fields = line.strip().split(",")
        team1fouls = int(fields[17])
        team2fouls = int(fields[33])
        if team1fouls < min_fouls:
            min_fouls = team1fouls
            min_team = fields[2]
            min_fields = fields    
        if team2fouls < min_fouls:
            min_fouls = team2fouls
            min_team = fields[18]
            min_fields = fields
    print(f"The fewest team fouls in a single game was {min_fouls} by {min_team} on {min_fields[1]}. The final score was {min_fields[2]} {min_fields[4]} - {min_fields[18]} {min_fields[20]}.")
    
if __name__ == "__main__":
    # largest_point_differential()
    # most_points_scored()
    fewest_team_fouls()