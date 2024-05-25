import cv2
import pandas as pd
import subprocess
import threading
import queue
import time
import numpy as np
from loguru import logger
import datetime
import csv
import matplotlib.pyplot as plt


# Function to run OpenFace subprocess
def run_openface(frame_path, output_dir):
    subprocess.run(
        [
            "docker",
            "exec",
            "-it",
            "openface",
            "FaceLandmarkImg",
            "-f",
            frame_path,
            "-out_dir",
            output_dir,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def convert_video(input_path, output_path, desired_fps):
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-vf",
        f"fps={desired_fps}",
        "-c:v",
        "h264",
        "-crf",
        "20",
        output_path,
    ]
    subprocess.run(cmd)


# Function to process frames and concatenate annotations
def process_frames(annotation_buffer, frame_rate, frame_queue, saccade_queue):
    # Buffer for storing annotations
    while True:
        try:
            # Get frame information from the queue
            frame_info = frame_queue.get(timeout=1)
            frame_path = frame_info["frame_path"]
            frame_index = frame_info["index"]
            frame_time = frame_info["frame_time"]

            # Run OpenFace subprocess
            run_openface(frame_path, "/tmp/openface/output")

            # Read the processed frame annotation
            processed_frame_annotation = pd.read_csv(
                f"/tmp/openface/output/frame_{frame_index}.csv"
            )

            # Add frame index and time to the annotation dataframe
            processed_frame_annotation["index"] = frame_info["index"]
            processed_frame_annotation["frame_time"] = frame_time

            # Append the annotation to the buffer
            annotation_buffer = pd.concat(
                [annotation_buffer, processed_frame_annotation], ignore_index=True
            )

            saccades = detect_saccades(
                annotation_buffer, current_index=frame_index, frame_rate=frame_rate
            )

            if frame_index % 40 == 0:
                annotation_buffer.to_csv(
                    "/tmp/openface/concatenated_annotations.csv", index=False
                )
                save_timestamps(annotation_buffer, all_saccades)
                plot_coordinates(annotation_buffer)

            # Put saccade information into the queue
            saccade_queue.put(saccades)

        except queue.Empty:
            continue


# Function to detect saccades
def detect_saccades(
    df, current_index, frame_rate, window_size=5, threshold_rate_of_change=0.4
):
    saccades = []

    # Calculate start and end indices of the window
    start_index = max(0, current_index - window_size + 1)
    end_index = current_index + 1

    # Calculate rate of change for gaze angle x and y within the window
    delta_angle_x = df["gaze_angle_x"].iloc[start_index:end_index].diff()
    delta_angle_y = df["gaze_angle_y"].iloc[start_index:end_index].diff()

    # Calculate time interval between frames (assuming uniform frame rate)
    time_interval = 1.0 / frame_rate

    # Calculate rate of change per second
    rate_of_change_x = delta_angle_x / time_interval
    rate_of_change_y = delta_angle_y / time_interval

    # Calculate moving average of rate of change using the specified window size
    moving_avg_rate_of_change_x = rate_of_change_x.rolling(
        window=window_size, min_periods=1
    ).mean()
    moving_avg_rate_of_change_y = rate_of_change_y.rolling(
        window=window_size, min_periods=1
    ).mean()

    # Check if rate of change exceeds threshold for any frame within the window
    saccade_indices = (moving_avg_rate_of_change_x.abs() > threshold_rate_of_change) | (
        moving_avg_rate_of_change_y.abs() > threshold_rate_of_change
    )
    saccades = df.iloc[start_index:end_index][saccade_indices].index.tolist()

    return saccades


def detect_saccades_avg(
    df, current_index, window_size=10, threshold_angle_x=0.1, threshold_angle_y=0.1
):
    saccades = []

    # Define the start and end index of the sliding window
    start_index = max(0, current_index - window_size // 2)
    end_index = min(len(df), current_index + window_size // 2 + 1)

    # Calculate average change in gaze angle over the window
    avg_delta_angle_x = np.mean(
        np.abs(df["gaze_angle_x"].iloc[start_index:end_index].diff().dropna())
    )
    avg_delta_angle_y = np.mean(
        np.abs(df["gaze_angle_y"].iloc[start_index:end_index].diff().dropna())
    )

    logger.info(avg_delta_angle_x)
    logger.info(avg_delta_angle_y)

    # Check if average change in gaze angle exceeds thresholds
    if avg_delta_angle_x > threshold_angle_x or avg_delta_angle_y > threshold_angle_y:
        saccades.append(current_index)

    return saccades


def save_timestamps(annotation_buffer, all_saccades):

    saccade_times = annotation_buffer[annotation_buffer["index"].isin(all_saccades)]

    # Convert frame times to human-readable format
    saccade_times["human_readable_time"] = saccade_times["frame_time"].apply(
        lambda ts: datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    )

    # Append timestamps to a text file
    with open("/tmp/openface/timestamps.txt", "a") as f:  # Append mode
        for index, row in saccade_times.iterrows():
            f.write(
                f"Index: {row['index']}, Frame Time: {row['human_readable_time']}\n"
            )


def plot_coordinates(annotation_buffer):
    time_indices = annotation_buffer["index"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot for gaze_0
    ax[0].plot(time_indices, annotation_buffer["gaze_0_x"], label="Gaze 0 X")
    ax[0].plot(time_indices, annotation_buffer["gaze_0_y"], label="Gaze 0 Y")
    ax[0].plot(time_indices, annotation_buffer["gaze_0_z"], label="Gaze 0 Z")
    ax[0].set_title("Left Eye Gaze Direction")
    ax[0].legend()

    # Plot for gaze_1
    ax[1].plot(time_indices, annotation_buffer["gaze_1_x"], label="Gaze 1 X")
    ax[1].plot(time_indices, annotation_buffer["gaze_1_y"], label="Gaze 1 Y")
    ax[1].plot(time_indices, annotation_buffer["gaze_1_z"], label="Gaze 1 Z")
    ax[1].set_title("Right Eye Gaze Direction")
    ax[1].legend()

    # Plot for combined gaze direction
    ax[2].plot(time_indices, annotation_buffer["gaze_0_x"], label="Gaze 0 X")
    ax[2].plot(time_indices, annotation_buffer["gaze_1_x"], label="Gaze 1 X")
    ax[2].plot(time_indices, annotation_buffer["gaze_0_y"], label="Gaze 0 Y")
    ax[2].plot(time_indices, annotation_buffer["gaze_1_y"], label="Gaze 1 Y")
    ax[2].plot(time_indices, annotation_buffer["gaze_0_z"], label="Gaze 0 Z")
    ax[2].plot(time_indices, annotation_buffer["gaze_1_z"], label="Gaze 1 Z")
    ax[2].set_title("Combined Eye Gaze Direction")
    ax[2].legend()

    ax[2].set_xlabel("Frame Index")

    plt.tight_layout()
    plt.savefig(f"/tmp/openface/gaze_plot_{frame_index}.png")
    plt.close(fig)


## initialize empty csv path every start
csv_file_path = "/tmp/openface/concatenated_annotations.csv"

# Open the CSV file in 'w' mode to create/truncate it
with open(csv_file_path, "w", newline="") as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)
    # Write an empty header row (assuming your CSV has headers)
    writer.writerow(["index", "frame_time", "annotation"])


all_saccades = []

import sys

## constant initializer
frame_queue = queue.Queue(maxsize=20)
saccade_queue = queue.Queue()

frame_index = 0
frame_rate = 5  # Set your desired FPS here
all_saccades = []


if len(sys.argv) > 1 and (sys.argv[1] == "-v" or sys.argv[1] == "--video"):
    if len(sys.argv) < 3:
        print("Error: Please provide a video file path after the flag.")
        sys.exit(1)

    input_video_path = sys.argv[2]
    output_video_path = "/tmp/converted_video.mp4"  # Output video file path

    # Convert video to the desired frame rate
    convert_video(input_video_path, output_video_path, frame_rate)

    # Open the converted video
    cap = cv2.VideoCapture(output_video_path)
else:
    cap = cv2.VideoCapture(0)

annotation_buffer = pd.DataFrame(columns=["index", "frame_time", "annotation"])


annotation_thread = threading.Thread(
    target=process_frames,
    args=(annotation_buffer, frame_rate, frame_queue, saccade_queue),
)
annotation_thread.start()

# Calculate the interval between frames in milliseconds
frame_interval = 1.0 / frame_rate

last_frame_time = time.time()


while True:
    current_time = time.time()

    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame if it's time to do so
    frame_path = f"/tmp/openface/frame_{frame_index}.jpg"
    cv2.imwrite(frame_path, frame)
    cv2.imshow("Original Frame", frame)

    frame_info = {
        "index": frame_index,
        "frame_time": current_time,
        "frame_path": frame_path,
    }
    frame_queue.put(frame_info)

    frame_index += 1
    last_frame_time = current_time

    # Process saccades if any
    while not saccade_queue.empty():
        saccades = saccade_queue.get()
        all_saccades.extend(saccades)
        logger.info(f"Saccades detected: {saccades}")

    # Check for user input to quit the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
