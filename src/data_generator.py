
import pandas as pd
import numpy as np

# Generate synthetic golf data
np.random.seed(42)

players = []

for player_id in range(1, 101):

    skill = np.random.uniform(0.5, 1.0)

    for shot_id in range(1, 101):

        club = np.random.choice(
            ["Driver", "Iron", "Wedge", "Putter"]
        )

        if club == "Driver":
            base_distance = 230
        elif club == "Iron":
            base_distance = 150
        elif club == "Wedge":
            base_distance = 80
        else:
            base_distance = 10

        wind_speed = np.random.uniform(0, 30)
        temperature = np.random.uniform(5, 35)

        distance = (
            base_distance * skill
            + np.random.normal(0, 10)
            - wind_speed * 0.3
        )

        accuracy = np.random.uniform(60, 100) * skill

        players.append({
            "player_id": player_id,
            "shot_id": shot_id,
            "club": club,
            "distance": round(max(distance, 0), 2),
            "accuracy": round(min(accuracy, 100), 2),
            "wind_speed": round(wind_speed, 2),
            "temperature": round(temperature, 2)
        })

df = pd.DataFrame(players)

df.to_csv("data/golf_data.csv", index=False)

print("Golf dataset generated successfully!")
print(df.head())
print("\nDataset shape:", df.shape)
