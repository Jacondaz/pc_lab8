#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace cv;

void processEyes(const Mat& gray, Mat& frame, Rect face, CascadeClassifier& eyesCascade) {
    std::vector<Rect> eyes;
    eyesCascade.detectMultiScale(gray(face), eyes, 3, 2, 0, Size(5, 5));

#pragma omp parallel for
    for (size_t j = 0; j < eyes.size(); ++j) {
        Rect eye = eyes[j];
        Point center(eye.x + eye.width / 2, eye.y + eye.height / 2);
        int radius = cvRound((eye.width + eye.height) * 0.25);
        circle(frame(face), center, radius, Scalar(63, 181, 110), 2);
    }
}

void processSmiles(const Mat& gray, Mat& frame, Rect face, CascadeClassifier& smileCascade) {
    std::vector<Rect> smiles;
    smileCascade.detectMultiScale(gray(face), smiles, 1.565, 30, 0, Size(30, 30));

#pragma omp parallel for
    for (size_t k = 0; k < smiles.size(); ++k) {
        Rect smile = smiles[k];
        rectangle(frame(face), smile, Scalar(181, 63, 169), 2);
    }
}

int main() {
    CascadeClassifier faceCascade, eyesCascade, smileCascade;

    if (!faceCascade.load("C:/Users/karet/source/repos/cv_practice/cv_practice/haara/haarcascade_frontalface_alt.xml") ||
        !eyesCascade.load("C:/Users/karet/source/repos/cv_practice/cv_practice/haara/haarcascade_eye_tree_eyeglasses.xml") ||
        !smileCascade.load("C:/Users/karet/source/repos/cv_practice/cv_practice/haara/haarcascade_smile.xml")) {
        std::cout << "Error" << std::endl;
        return -1;
    }

    VideoCapture cap("C:/Users/karet/Downloads/face.mp4");

    if (!cap.isOpened()) {
        std::cout << "Error!" << std::endl;
        return -1;
    }

    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);

    VideoWriter videoWriter("output.mp4", VideoWriter::fourcc('X', 'V', 'I', 'D'), 20, Size(frame_width, frame_height));

    auto start = std::chrono::steady_clock::now();

    Mat frame, gray;
    std::vector<Rect> faces, eyes, smiles;

    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cout << "End of video" << std::endl;
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        std::vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 2, 3, 0, Size(20, 20));

#pragma omp parallel for
        for (size_t i = 0; i < faces.size(); ++i) {
            Rect face = faces[i];
            rectangle(frame, face, Scalar(58, 64, 224), 2);

            processEyes(gray, frame, face, eyesCascade);
            processSmiles(gray, frame, face, smileCascade);

            blur(frame(face), frame(face), Size(3, 3));
        }

        imshow("Video", frame);
        videoWriter.write(frame);

        if (waitKey(25) == 'q') {
            break;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "The time spent on the program: " << elapsed_seconds.count() << " seconds" << std::endl;

    videoWriter.release();
    destroyAllWindows();
    return 0;
}
