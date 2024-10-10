// I have only included the OpenCV library that contains DNN module for inference.
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// Standart IO and reading libraries are included. 
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;

// Threshold values. CONFIDENCE_THRESHOLD is applied to filter the inference frames that have been detected poorly.
// NMS_THRESHOLD is used in Non-Maxima Suppression to eliminate overlapping bounding boxes, keeping only the box with the highest confidence score.
const float CONFIDENCE_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.3f;

// Draws bounding boxes on the input image for the objects detected by the YOLO algorithm.
// Takes the input frame, output data, and the class list that were used during the training process.
void PostProcess(Mat& frame, const std::vector<Mat>& outputs, const std::vector<std::string>& classList) {
    std::vector<int> classIDs;
    std::vector<float> confidences;
    std::vector<Rect> boxes;

    // Resizing factors (adjust these values according to the pixel resolution used during model training)
    float xFactor = frame.cols / 416.0;
    float yFactor = frame.rows / 416.0;

    // Each object in the 'outputs' vector contains the x and y coordinates of the box's center (data[0] and data[1]), 
    // the width and height values of the bounding box (data[2] and data[3]), and the confidence score of that detection (data[4]).
    // The 'cv::Mat::data' function returns a pointer to the data as an unsigned char type, which we need to cast to a float pointer 
    // to properly access these variables as floating-point numbers.
    float* data = (float*)outputs[0].data;

    // This value depends on the YOLO model architecture and the input image size.
    // For example, in YOLOv5, this number is calculated based on the grid size 
    // at multiple scales (small, medium, and large) used for object detection.
    const int rows = 25200; 

    // Iterate over each detection
    for (int i = 0; i < rows; ++i) {
        // Gets the confidence value of the bounding box data.
        float confidence = data[4];
        
        // If the confidence is high enough to pass the threshold, the class scores are stored in the 'scores' object as a single row, 
        // with a column for each class in the model.
        // The 'cv::minMaxLoc()' function writes the minimum score value to the second argument and the maximum score value to the third argument.
        // It also writes the location of the minimum score to the fourth argument and the location of the maximum score to the fifth argument 
        // as a row and column index (in a 'cv::Point' object).
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float* classScores = data + 5;
            Mat scores(1, classList.size(), CV_32FC1, classScores);
            Point classIDpoint;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &classIDpoint);

            if (maxClassScore > CONFIDENCE_THRESHOLD) {
                // Coordinate values of the bounding box's center point.
                // In computer vision, the x-coordinate typically increases from left to right, while the y-coordinate increases from top to bottom.
                // This means that moving right increases the x value, and moving down increases the y value.
                float x = data[0];
                float y = data[1];
                // Width and height values of the bounding box
                float w = data[2];
                float h = data[3];

                // To create a proper rectangle object in OpenCV, we need to calculate the top-left corner's coordinate values.
                // This is done by determining the position of the top-left corner based on the bounding box's center point.
                // Additionally, we must scale the rectangle dimensions back to the original image size, as the coordinates were initially calculated on a (416x416) px resized image.
                int left = int((x - 0.5 * w) * xFactor);
                int top = int((y - 0.5 * h) * yFactor);
                int width = int(w * xFactor);
                int height = int(h * yFactor);

                // Adds the bounding boxes, confidence scores, and class IDs to their respective vectors.
                boxes.emplace_back(Rect(left, top, width, height));
                confidences.emplace_back(confidence);
                classIDs.emplace_back(classIDpoint.x);

            }
        }
               
        // The 'data' pointer is incremented in each iteration by the size of the 'classList' plus a fixed value of 5. 
        // Each element in the 'outputs' vector represents a detected object and contains the following values:
        // (x, y, w, h, confidence, and class score values). 
        // The first 5 parameters (x, y, w, h, confidence) are fixed and are always present regardless of the model.
        // The number of class score values depends on the number of classes defined in the 'classList' and varies with each project.
        data += 5 + classList.size(); // 5: x, y, w, h, confidence
        
    }

    // Applies Non-Maxima Suppression (NMS) to remove redundant overlapping bounding boxes. 
    // NMS helps in keeping only the most relevant bounding boxes by comparing the confidence scores of the overlapping boxes.
    // Boxes with lower confidence scores are eliminated if they significantly overlap with a box that has a higher confidence score.
    // The resulting indices of the retained boxes (those with the best confidence scores) are stored in the 'indices' vector.
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        int classID = classIDs[idx];

        // Draws the bounding box on the detected object in the frame
        rectangle(frame, box, Scalar(240, 32, 160), 2);

        // Creates a label with the class name and confidence score for the detected object
        std::string label = classList[classID] + ": " + std::to_string(confidences[idx]);

        // Calculates the size of the label text to properly display it above the bounding box
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        // Draws a filled rectangle to serve as a background for the label text above the bounding box
        rectangle(frame, Point(box.x, box.y - labelSize.height - baseLine),
        Point(box.x + labelSize.width, box.y), Scalar(255, 0, 0), FILLED);

        // Renders the label text on top of the filled rectangle above the bounding box
        putText(frame, label, Point(box.x, box.y - baseLine), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1 ,16);
    }
}

void LoadClassNames(const std::string& filePath, std::vector<std::string>& classList) {
    std::ifstream file(filePath);
    std::string line;

    // Check if the file was successfully opened for reading.
    // If the file could not be opened, display an error message with the file path and terminate the loading process.
    // This ensures that the program does not attempt to read from a file that is unavailable or inaccessible.
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file: " << filePath << std::endl;
        return;  // Exit the function if the file could not be opened
    }


    // Reads the file line by line, storing each non-empty line into the classList vector.
    // This ensures that only valid class names are added, skipping any empty lines.
    while (std::getline(file, line)) {
        if (!line.empty()) {  // Check to avoid adding empty lines to the classList
            classList.push_back(line);  // Add the class name to the vector
        }
    }

    file.close();
}

//!Please change your path relative to your ONNX model location.
const std::string modelPath = "source/inference/bestModel.onnx";
// Please change your path relative to your image location.
const std::string imagePath = "source/inference/inferenceImage.jpg";
// Please change your path relative to your class list TXT location.
const std::string classListPath = "source/inference/classList.txt";


int main() {
    // Creates the class list
    std::vector<std::string> classList;

    // Loads the class names from a specified TXT file into the classList vector.
    // If your model only uses a few classes, you may directly add class names using 'std::vector::push_back()'.
    LoadClassNames(classListPath, classList);

    // Loads the image.
    Mat frame = imread(imagePath);
    if (frame.empty()) {
        std::cerr << "Image not found! Please check the image path." << std::endl;
        return -1;
    }

    // Loads YOLOv5 model. 
    Net net = readNet(modelPath);
    if (net.empty()) {
        std::cerr << "Model not found! Please check the model path." << std::endl;
        return -1;
    }

    // Pre-processes the image by converting it into a blob format suitable for deep learning models.
    // This step includes resizing the image to the desired input size (416x416), normalizing pixel values by scaling them to the range [0, 1],
    // and applying mean subtraction if necessary. The blob serves as the input for the neural network.
    Mat blob = blobFromImage(frame, 1.0 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);

    // Sets the input for the neural network to the pre-processed blob.
    // This ensures that the neural network receives correctly formatted image data for inference.
    net.setInput(blob);

    // Performs a forward pass through the neural network to carry out the inference.
    // The resulting detections (outputs) are stored in a vector of matrices representing different layer's outputs.
    std::vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Post-processes the network's outputs to extract and visualize detected objects in the image.
    // This involves filtering detections, drawing bounding boxes, and adding labels to the detected objects.
    PostProcess(frame, outputs, classList);

    // Displays the image with detected objects.
    // The image remains open until a key is pressed by the user.
    imshow("YOLO Object Detection", frame);

    // Waits for a key press indefinitely, effectively pausing the program execution until a key is pressed.
    waitKey(0);


    return 0;
}
