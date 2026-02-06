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
    
    
if __name__ == "__main__":
    largest_point_differential()
    most_points_scored()