#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <ctime>

// Function to add salt and pepper noise to an image
void addSaltAndPepperNoise(cv::Mat &image, double noise_ratio) {
    int rows = image.rows;
    int cols = image.cols;
    int ch = image.channels();
    int num_of_noise_pixels = static_cast<int>((rows * cols * ch) * noise_ratio);

    for (int i = 0; i < num_of_noise_pixels; i++) {
        int r = std::rand() % rows;
        int c = std::rand() % cols;
        int _ch = std::rand() % ch;

        uchar* pixel = image.ptr<uchar>(r) + (c * ch) + _ch;
        *pixel = (i % 2 == 0) ? 255 : 0; // Alternating between salt and pepper
    }
}

int main() {
    // Initialize random seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Read the image
    std::string filename = "resources/image1.jpg"; // Replace with your image path
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Error: Image cannot be loaded!" << std::endl;
        return -1;
    }

    // Add salt and pepper noise
    double noise_ratio = 0.05; // Adjust the ratio as needed
    addSaltAndPepperNoise(image, noise_ratio);

    // Save the noisy image
    std::string outputFilename = "resources/noisy_image1.jpg";
    cv::imwrite(outputFilename, image);

    return 0;
}
