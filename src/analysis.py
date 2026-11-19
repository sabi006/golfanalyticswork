
import pandas as pd

# Load golf dataset
df = pd.read_csv("data/golf_data.csv")

print("========== GOLF DATA ANALYSIS ==========")

# 1. General information
print("\n1. GENERAL INFORMATION")
print("Total shots:", len(df))
print("Total players:", df["player_id"].nunique())

# 2. Average statistics
print("\n2. OVERALL STATISTICS")
print("Average distance:", round(df["distance"].mean(), 2), "m")
print("Average accuracy:", round(df["accuracy"].mean(), 2), "%")

# 3. Club analysis
print("\n3. CLUB ANALYSIS")

club_stats = df.groupby("club").agg({
    "distance": "mean",
    "accuracy": "mean"
}).round(2)

print(club_stats)

# 4. Best players
print("\n4. TOP 10 PLAYERS")

player_stats = df.groupby("player_id").agg({
    "distance": "mean",
    "accuracy": "mean"
}).round(2)

top_players = player_stats.sort_values(
    by="accuracy",
    ascending=False
).head(10)

print(top_players)

# 5. Weather analysis
print("\n5. WEATHER ANALYSIS")

print("Average wind speed:",
      round(df["wind_speed"].mean(), 2), "km/h")

print("Average temperature:",
      round(df["temperature"].mean(), 2), "°C")

print("\nAnalysis completed successfully!")
