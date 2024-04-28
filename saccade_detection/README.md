## OpenFace Eye Saccade Detector

Experiment run on: Macbook 13 Pro M1

### Motivations and Findings:
- M1 has limited support in terms of some of the C++ library, hence, utilizing docker to run the experiment makes it more environment agnostic
- Treating it as running a docker container as a separate service
- Image input streamed from webcam and exposing some ports and edit access to the tmp directory such that it can be consumed by other services
- Changes are made to CMakeLists.txt and FindOpenBLAS.cmake to enable compiling in Docker

### Script:
- Takes in input from normal webcam stream, since saccades are time sensitive, low sleep time is set in between frame interval (shown in code)
- Multiple threads are used for data buffering, processing, and storing of iamge annotation (check the process_frames function)

## Overall Workflow
- Running Webcam as an input
- A separate thread is reading through the image input (allocated in queue), run openface (via subprocess)
- Once openface detection subprocess is completed, frame annotation is buffered in the memory (save to csv for further processing every n frames)
- The buffer result is passed to detect saccades function, whereby running sliding window (window size can be customized), to process the gaze_x and gaze_y angle change (averaged and thresholded within the defined value)
- Once detected, update frame number in the queue (pointing that saccade took place)
