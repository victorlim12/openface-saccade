import cv2
import pandas as pd
import subprocess
import threading
import queue
import time
import numpy as np
from loguru import logger


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
        ]
    )


# Function to process frames and concatenate annotations
def process_frames(frame_queue, saccade_queue):
    # Buffer for storing annotations
    annotation_buffer = pd.DataFrame(columns=["index", "frame_time", "annotation"])

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

            saccades = detect_saccades(annotation_buffer, current_index=frame_index)

            if frame_index % 40 == 0:
                annotation_buffer.to_csv(
                    "/tmp/openface/concatenated_annotations.csv", index=False
                )

            # Put saccade information into the queue
            saccade_queue.put(saccades)

        except queue.Empty:
            continue


# Function to detect saccades
def detect_saccades(
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


cap = cv2.VideoCapture(0)

frame_queue = queue.Queue(maxsize=20)
saccade_queue = queue.Queue()

annotation_thread = threading.Thread(
    target=process_frames, args=(frame_queue, saccade_queue)
)
annotation_thread.start()

frame_index = 0
saccade_fps = 30
saccade_delay_ms = int(1000 / saccade_fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_path = f"/tmp/openface/frame_{frame_index}.jpg"
    cv2.imwrite(frame_path, frame)

    cv2.imshow("Original Frame", frame)

    frame_time = time.time()

    frame_info = {
        "index": frame_index,
        "frame_time": frame_time,
        "frame_path": frame_path,
    }
    frame_queue.put(frame_info)

    frame_index += 1

    time.sleep(saccade_delay_ms / 1000.0)
    while not saccade_queue.empty():
        saccades = saccade_queue.get()
        logger.info(f"Saccades detected: {saccades}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
