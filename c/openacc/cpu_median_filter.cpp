#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

// Function to apply median filter to a single channel
void applyMedianFilter(const cv::Mat& input, cv::Mat& output, int kernelSize) {
    int edge = kernelSize / 2;
    std::cout << "Inside Medail Fitler\n";
    for (int y = edge; y < input.rows - edge; y++) {
        for (int x = edge; x < input.cols - edge; x++) {
           // std::cout << x << " " << y << "\n";
            std::vector<uchar> neighbors;
            for (int dy = -edge; dy <= edge; dy++) {
                for (int dx = -edge; dx <= edge; dx++) {
                    neighbors.push_back(input.at<uchar>(y + dy, x + dx));
                }
            }
            std::nth_element(neighbors.begin(), neighbors.begin() + neighbors.size() / 2, neighbors.end());
            output.at<uchar>(y, x) = neighbors[neighbors.size() / 2];
        }
    }
}

int main() {
    cv::Mat image = cv::imread("resources/image1.jpg", cv::IMREAD_COLOR);


    cv::Mat frame, filteredFrame;
    cv::Mat channels[3], outputChannels[3];
    int N = 1; // Replace with the number of frames you want to process



        // Save the frame before filtering
        std::string filenameBefore = "before.jpg" ;
        cv::imwrite(filenameBefore, image);



        // Split the frame into its color channels
        cv::split(image, channels);

        // Apply median filter to each channel
        double t0 = static_cast<double>(cv::getTickCount());
        for (int i = 0; i < 3; i++) {
            outputChannels[i] = cv::Mat(image.rows, image.cols, CV_8UC1);
            applyMedianFilter(channels[i], outputChannels[i], 5); // Kernel size is 5
        }
        double t1 = static_cast<double>(cv::getTickCount());
    double elapsed = (t1 - t0) / cv::getTickFrequency();
    std::cout << "Time elapsed: " << elapsed << " seconds." << std::endl;

        // Merge the channels back
        cv::merge(outputChannels, 3, filteredFrame);


        // Save the frame after filtering
        std::string filenameAfter = "after.jpg";
        cv::imwrite(filenameAfter, filteredFrame);
    


    return 0;
}