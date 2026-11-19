
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import subprocess
import sys

from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "golf_model.pkl"
DATA_PATH = BASE_DIR / "data" / "golf_data.csv"
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.csv"
CHARTS_PATH = BASE_DIR / "charts"

VISUALIZATION_SCRIPT = BASE_DIR / "src" / "visualization.py"


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Golf Analytics",
    page_icon="⛳",
    layout="wide"
)


# ==========================================
# 3. LOAD MODEL AND DATA
# ==========================================

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)


# ==========================================
# 4. LOAD PREDICTION HISTORY
# ==========================================

if "predictions" not in st.session_state:

    if PREDICTIONS_PATH.exists():

        saved_predictions = pd.read_csv(PREDICTIONS_PATH)

        st.session_state.predictions = saved_predictions.to_dict(
            "records"
        )

    else:

        st.session_state.predictions = []


# ==========================================
# 5. APPLICATION TITLE
# ==========================================

st.title("⛳ Golf Analytics Dashboard")

st.write(
    "Golf Shot Analysis and Distance Prediction"
)

st.divider()


# ==========================================
# 6. DASHBOARD STATISTICS
# ==========================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Shots",
    f"{len(df):,}"
)

col2.metric(
    "Players",
    df["player_id"].nunique()
)

col3.metric(
    "Average Distance",
    f"{df['distance'].mean():.2f} m"
)

st.divider()


# ==========================================
# 7. APPLICATION TABS
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "Distance Prediction",
    "Data Visualization",
    "Dataset"
])


# ==========================================
# TAB 1: DISTANCE PREDICTION
# ==========================================

with tab1:

    st.header("Predict Golf Shot Distance")

    st.write(
        "Enter your shot characteristics below."
    )

    col1, col2 = st.columns(2)

    with col1:

        club = st.selectbox(
            "Select Golf Club",
            ["Driver", "Iron", "Wedge", "Putter"]
        )

        accuracy = st.slider(
            "Shot Accuracy (%)",
            min_value=0,
            max_value=100,
            value=75
        )

    with col2:

        wind_speed = st.slider(
            "Wind Speed (km/h)",
            min_value=0,
            max_value=30,
            value=10
        )


    # Make prediction
    if st.button("Predict Distance", type="primary"):

        input_data = pd.DataFrame({
            "club": [club],
            "accuracy": [accuracy],
            "wind_speed": [wind_speed]
        })

        prediction = float(
            model.predict(input_data)[0]
        )

        # Save prediction in session history
        st.session_state.predictions.append({

            "club": club,
            "accuracy": accuracy,
            "wind_speed": wind_speed,
            "distance": prediction

        })

        # Save history to CSV
        history = pd.DataFrame(
            st.session_state.predictions
        )

        history.to_csv(
            PREDICTIONS_PATH,
            index=False
        )

        # Regenerate charts
        subprocess.run(
            [
                sys.executable,
                str(VISUALIZATION_SCRIPT)
            ],
            cwd=BASE_DIR,
            check=True
        )

        st.success("Prediction completed!")

        st.metric(
            "Predicted Shot Distance",
            f"{prediction:.2f} meters"
        )


    # ======================================
    # PREDICTION HISTORY
    # ======================================

    st.divider()

    st.subheader("Prediction History")

    if st.session_state.predictions:

        history = pd.DataFrame(
            st.session_state.predictions
        )

        st.dataframe(
            history,
            use_container_width=True
        )

        # ==================================
        # PREDICTIONS VS HISTORICAL DATA
        # ==================================

        st.subheader(
            "Predicted Distance vs Wind Speed"
        )

        fig, ax = plt.subplots(figsize=(10, 5))

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
        ax.scatter(
            history["wind_speed"],
            history["distance"],
            color="red",
            marker="*",
            s=250,
            edgecolors="black",
            linewidths=0.8,
            label="Your Predictions",
            zorder=5
        )

        ax.set_title(
            "Golf Shot Distance: Historical Data vs Predictions"
        )

        ax.set_xlabel("Wind Speed (km/h)")
        ax.set_ylabel("Distance (meters)")

        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig)

        plt.close(fig)


        # ==================================
        # CLEAR HISTORY
        # ==================================

        if st.button("Clear Prediction History"):

            st.session_state.predictions = []

            empty_history = pd.DataFrame(
                columns=[
                    "club",
                    "accuracy",
                    "wind_speed",
                    "distance"
                ]
            )

            empty_history.to_csv(
                PREDICTIONS_PATH,
                index=False
            )

            subprocess.run(
                [
                    sys.executable,
                    str(VISUALIZATION_SCRIPT)
                ],
                cwd=BASE_DIR,
                check=True
            )

            st.rerun()

    else:

        st.info(
            "No predictions yet. Make your first prediction!"
        )


# ==========================================
# TAB 2: DATA VISUALIZATION
# ==========================================

with tab2:

    st.header("Golf Data Visualization")

    charts = [

        (
            "Average Distance by Golf Club",
            "club_distance.png"
        ),

        (
            "Golf Shot Accuracy Distribution",
            "accuracy_distribution.png"
        ),

        (
            "Wind Speed vs Shot Distance",
            "wind_distance.png"
        )

    ]

    for title, filename in charts:

        chart_path = CHARTS_PATH / filename

        st.subheader(title)

        if chart_path.exists():

            st.image(
                chart_path.read_bytes(),
                use_container_width=True
            )

        else:

            st.warning(
                f"Chart not found: {filename}"
            )


# ==========================================
# TAB 3: DATASET
# ==========================================

with tab3:

    st.header("Golf Dataset")

    st.write(
        "Explore the generated synthetic golf data."
    )

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    st.write(
        "Dataset shape:",
        df.shape
    )

    st.write("Missing values:")

    st.dataframe(
        df.isnull().sum()
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe()
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Golf Analytics Project | Machine Learning Application"
)
