// Include necessary headers from OpenCV library for video processing and standard input/output operations
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    // Ensure the correct number of command line arguments are passed
    if (argc != 4) {
        // If not, print usage information and exit
        std::cerr << "Usage: " << argv[0] << " <path_to_video> <base_memory_address> <output_video_path>" << std::endl;
        return -1;
    }

    // Store command line arguments in variables for easier access
    std::string videoPath = argv[1];
    std::string baseMemoryAddress = argv[2];
    std::string outputPath = argv[3];

    // Display the base memory address for confirmation or debugging purposes
    std::cout << "Base Memory Address: " << baseMemoryAddress << std::endl;

    // Open the video file specified by the user
    cv::VideoCapture video(videoPath);

    // Check if the video file was successfully opened
    if (!video.isOpened()) {
        std::cerr << "Failed to open video file." << std::endl;
        return -1;
    }

    // Initialize variables for video processing
    cv::Mat frame; // Matrix to store each frame of the video
    double scalingFactor = 0.75; // Factor to scale down the video frames by to reduce processing time
    // Calculate scaled frame dimensions
    int frameWidth = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH) * scalingFactor);
    int frameHeight = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT) * scalingFactor);
    double fps = video.get(cv::CAP_PROP_FPS); // Frames per second of the video

    // Setup the output video writer with MJPG codec, same fps as input video, and scaled frame size
    cv::VideoWriter output(outputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frameWidth, frameHeight), true);

    // Create a background subtractor object for motion detection
    auto backgroundSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, true);

    // Process each frame of the video
    while (true) {
        // Read a new frame from the video
        if (!video.read(frame)) {
            std::cerr << "Failed to read frame from video source." << std::endl;
            break; // Exit loop if there are no more frames
        }

        // Resize the frame according to the scaling factor
        cv::resize(frame, frame, cv::Size(frameWidth, frameHeight));

        // Apply background subtraction to get the foreground mask
        cv::Mat fgMask;
        backgroundSubtractor->apply(frame, fgMask, 0.01);

        // Improve mask with morphological operations
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

        // Find contours in the foreground mask
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Draw bounding boxes around detected contours
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 100) continue; // Ignore small contours
            cv::Rect boundingBox = cv::boundingRect(contour);
            cv::rectangle(frame, boundingBox, cv::Scalar(0, 255, 0), 2); // Draw green rectangle
        }

        // Display the processed video frame with detected objects
        cv::imshow("Video feed", frame);

        // Write the processed frame to the output video
        output.write(frame);

        // Break the loop if any key is pressed
        if (cv::waitKey(1) >= 0) break;
    }

    // Release resources and close video files and windows
    output.release();
    video.release();
    cv::destroyAllWindows();

    return 0;
}
