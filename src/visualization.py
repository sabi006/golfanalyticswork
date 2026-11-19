
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "golf_data.csv"
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.csv"
CHARTS_PATH = BASE_DIR / "charts"

# Create charts folder if it does not exist
CHARTS_PATH.mkdir(parents=True, exist_ok=True)

# Load historical dataset
df = pd.read_csv(DATA_PATH)

# Load prediction history
if PREDICTIONS_PATH.exists():
    predictions = pd.read_csv(PREDICTIONS_PATH)
else:
    predictions = pd.DataFrame(
        columns=["club", "accuracy", "wind_speed", "distance"]
    )


# 1. Average distance by golf club
club_distance = df.groupby("club")["distance"].mean()

plt.figure(figsize=(9, 5))

club_distance.plot(kind="bar")

plt.title("Average Distance by Golf Club")
plt.xlabel("Club")
plt.ylabel("Distance (meters)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(CHARTS_PATH / "club_distance.png")
plt.close()


# 2. Accuracy distribution
plt.figure(figsize=(9, 5))

plt.hist(
    df["accuracy"],
    bins=20,
    edgecolor="black"
)

plt.title("Golf Shot Accuracy Distribution")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Shots")

plt.tight_layout()
plt.savefig(CHARTS_PATH / "accuracy_distribution.png")
plt.close()


# 3. Wind speed vs shot distance
fig, ax = plt.subplots(figsize=(10, 6))

# Historical data
ax.scatter(
    df["wind_speed"],
    df["distance"],
    color="skyblue",
    alpha=0.15,
    s=12,
    label="Historical Data"
)

# New predictions
if not predictions.empty:

    ax.scatter(
        predictions["wind_speed"],
        predictions["distance"],
        color="red",
        marker="*",
        s=250,
        edgecolors="black",
        linewidths=0.8,
        label="Your Predictions",
        zorder=5
    )

ax.set_title("Golf Shot Distance: Historical Data vs Predictions")
ax.set_xlabel("Wind Speed (km/h)")
ax.set_ylabel("Distance (meters)")

ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(CHARTS_PATH / "wind_distance.png")
plt.close()

print("All charts generated successfully!")
