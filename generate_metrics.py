import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set Streamlit page configuration
st.set_page_config(page_title="Metrics Dashboard", layout="wide")

# Initialize global metrics storage
metrics_log = {
    "frame": [],
    "fps": [],
    "precision": [],
    "recall": [],
    "speed_error_mae": [],
    "speed_error_rmse": []
}

# Function to log metrics
def log_metrics(frame, fps, precision, recall, ground_truth_speeds, estimated_speeds):
    # Calculate speed estimation errors
    mae = mean_absolute_error(ground_truth_speeds, estimated_speeds)
    rmse = np.sqrt(mean_squared_error(ground_truth_speeds, estimated_speeds))

    # Append to metrics log
    metrics_log["frame"].append(frame)
    metrics_log["fps"].append(fps)
    metrics_log["precision"].append(precision)
    metrics_log["recall"].append(recall)
    metrics_log["speed_error_mae"].append(mae)
    metrics_log["speed_error_rmse"].append(rmse)

    # Print logged metrics (for debugging)
    print(f"Frame: {frame}, FPS: {fps:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, "
          f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Streamlit layout for real-time metric display
st.title("Real-Time Metrics Dashboard")
st.write("This dashboard displays system performance metrics such as FPS, detection precision, recall, and speed estimation accuracy.")

# Create columns for displaying real-time metrics
col_fps, col_precision, col_recall, col_mae, col_rmse = st.columns(5)
fps_display = col_fps.metric("FPS", 0)
precision_display = col_precision.metric("Precision", 0.0)
recall_display = col_recall.metric("Recall", 0.0)
mae_display = col_mae.metric("Speed MAE (km/h)", 0.0)
rmse_display = col_rmse.metric("Speed RMSE (km/h)", 0.0)

# Placeholder for data visualization
visualization_container = st.empty()

# Example function to simulate video frame processing
def process_video():
    frame_count = 0
    total_time = 0

    # Simulated ground truth and predicted speeds
    ground_truth_speeds = [50, 60, 70, 40]  # Example ground truth speeds
    estimated_speeds = [52, 58, 73, 39]    # Example predicted speeds

    for frame in range(1, 101):  # Simulate 100 frames
        start_time = time.time()

        # Simulate precision and recall
        precision = np.random.uniform(0.85, 0.95)
        recall = np.random.uniform(0.80, 0.90)

        # Simulate processing delay
        time.sleep(0.03)  # Simulating ~30 FPS
        end_time = time.time()

        # Calculate FPS
        fps = 1 / (end_time - start_time)
        frame_count += 1
        total_time += (end_time - start_time)

        # Log metrics
        log_metrics(frame, fps, precision, recall, ground_truth_speeds, estimated_speeds)

        # Update Streamlit real-time displays
        fps_display.metric("FPS", f"{fps:.2f}")
        precision_display.metric("Precision", f"{precision:.2f}")
        recall_display.metric("Recall", f"{recall:.2f}")
        mae_display.metric("Speed MAE (km/h)", f"{metrics_log['speed_error_mae'][-1]:.2f}")
        rmse_display.metric("Speed RMSE (km/h)", f"{metrics_log['speed_error_rmse'][-1]:.2f}")

        # Periodically update the visualization
        if frame % 10 == 0:  # Update every 10 frames
            visualize_metrics()

# Function to visualize logged metrics
def visualize_metrics():
    df_metrics = pd.DataFrame(metrics_log)

    with visualization_container.container():
        st.subheader("Metrics Over Time")

        # Line chart for FPS
        st.line_chart(df_metrics[["frame", "fps"]].set_index("frame"), height=200, use_container_width=True)

        # Line chart for precision and recall
        st.line_chart(df_metrics[["frame", "precision", "recall"]].set_index("frame"), height=200, use_container_width=True)

        # Line chart for MAE and RMSE
        st.line_chart(df_metrics[["frame", "speed_error_mae", "speed_error_rmse"]].set_index("frame"), height=200, use_container_width=True)

        st.write("**Metrics Data Table**")
        st.dataframe(df_metrics)

# Process video simulation
if st.button("Start Processing"):
    process_video()
