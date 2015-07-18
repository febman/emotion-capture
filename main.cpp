#include <opencv2/opencv.hpp>
#include <stasm_lib.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <libsvm/svm.h>
#include <vector>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#define kNumClasses 6
#define kNumAttr 154

using namespace cv;

typedef enum {NEUTRAL, ANGRY, CONTEMPT, DISGUST, FEAR, HAPPY, SAD,
              SURPRISE} emotion_type;
// double sample[] = {0.3456,0.501736,0.637789,0.434131,0.67091,0.506209,0.774421,0.57365,0.801669,0.369103,0.63683,0.427438,0.557188,0,0.489473,0.330205,0.49362,0.432366,0.478647,0.625093,0.610607,0.502084,0.498532,0.442381,0.599881,0.468418,0.523966,0.774152,0.542852,0,0.532809,0.782858,0.35473,0.338028,0.383349,0.25971,0.339092,0.35287,0.290431,0.32713,0.33519,0.365277,0.358864,0.429843,0.297704,0.372112,0.309986,0.331853,0.364021,0.22099,0.460859,0.329719,0.391221,0.30281,0.314147,0.334992,0.402591,0.555468,0.393714,0.531956,0.383657,0.611547,0.401101,0.644529,0.399358,0.686666,0.396547,0.632823,0.340352,0.558175,0.360233,0.603732,0.37594,0.648469,0.378973,0.64862,0.460551,0.632953,0.501103,0.632953,0.362779,0.655531,0.396638,0.663212,0.413725,0.660684,0.461478,0.656263,0.443778,0.57905,0.472049,0.592845,0.430782,0.619845,0.395356,0.652497,0.363975,0.547149,0.333895,0.556984,0.342175,0.579259,0.32885,0.623029,0.383779,0.631062,0.516293,0.5802,0.500714,0.430809,0.528475,0.508128,0.452684,0.605384,0.250261,0.564742,0.353824,0.436609,0.380734,0.570744,0.392296,0.583438,0.419357,0.605971,0.446168,0.649736,0.495483,0.586518,0.468518,0.563903,0.532263,0.527867,0.5158,0.638589,0.514193,0.682893,0.438192,0.682932,0.42495,0.671373,0.517377,0.663978,0.500746,0.642687,0.555317,0.609606,0.59819,0.673121,0.595583,0.703065,0.407841,0.710612,0.34297,0.695015};

int vec_mode(std::vector<int> &vec)
{
    std::vector<int> histogram(8,0);
    for (const int& el : vec) {
        ++histogram[el];
    }
    return std::max_element(histogram.begin(), histogram.end() ) - histogram.begin();
}

void rotate_and_scale_landmarks(float *landmarks)
{
    cv::Point eye1 = cv::Point(landmarks[38*2], landmarks[38*2+1]);
    cv::Point eye2 = cv::Point(landmarks[39*2], landmarks[39*2+1]);
    float rot = std::atan2(-eye2.y+eye1.y, eye2.x-eye1.x);

    float min_x = 1e6;
    float min_y = 1e6;
    float max_x = -1e6;
    float max_y = -1e6;

    for (int i = 0; i < stasm_NLANDMARKS; i++) {
        cv::Point currPt = cv::Point(landmarks[i*2], landmarks[i*2+1]);
        float x = currPt.x - eye1.x;
        float y = currPt.y - eye1.y;
        float x_new = eye1.x + std::cos(rot)*x - std::sin(rot)*y;
        float y_new = eye1.y + std::sin(rot)*x + std::cos(rot)*y;
        landmarks[i*2] = x_new;
        landmarks[i*2+1] = y_new;
        min_x = x_new < min_x ? x_new : min_x;
        min_y = y_new < min_y ? y_new : min_y;
        max_x = x_new > max_x ? x_new : max_x;
        max_y = y_new > max_y ? y_new : max_y;
    }

    float width = max_x - min_x;
    float height = max_y - min_y;

    for (int i = 0; i < stasm_NLANDMARKS; i++) {
        landmarks[i*2] = (landmarks[i*2] - min_x)/width;
        landmarks[i*2+1] = (landmarks[i*2+1] - min_y)/height;
    }
}

int main(int argc, char **argv)
{
    double mins[kNumAttr] = {0.0};
    double ranges[kNumAttr] = {0.0};
    // std::ifstream minFile("min_vals.csv");
    // std::ifstream maxFile("max_vals.csv");
    // std::string line;

    // int cnt = 0;
    // while(std::getline(minFile, line)) {
    //     mins[cnt] = std::atof(line.c_str());
    //     cnt++;
    // }

    // cnt = 0;
    // while(std::getline(maxFile, line)) {
    //     ranges[cnt] = std::atof(line.c_str()) - mins[cnt];
    //     if (ranges[cnt] == 0) ranges[cnt] = 1;
    //     cnt++;
    // }

    bool neutralFrameCaptured = false;
    float landmarks[2 * stasm_NLANDMARKS];
    int foundface;
    std::vector<int> predictions {};

    double prob_estimates[kNumClasses] = {0.0};
    int labels[kNumClasses] = {0};
    struct svm_model *model = svm_load_model("training_data.model");
    svm_get_labels(model, labels);
    struct svm_node detection_features[kNumAttr+1];

    cv::VideoCapture cap(0);
    if(!cap.isOpened()) {
        return -1;
    }

    cv::Mat frame;
    cv::Mat_<unsigned char> grayFrame;



    for(;;) {
        cap >> frame; // get a new frame from camera
        cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);

        if (!stasm_search_single(&foundface, landmarks,
                                 (const char*)grayFrame.data, grayFrame.cols, grayFrame.rows,
                                 "", "data")) {
            printf("Error in stasm_search_single: %s\n", stasm_lasterr());
            continue;
        }

        if (!foundface) {
            printf("No face found\n");
            continue;
        }
        else {
            rotate_and_scale_landmarks(landmarks);
        }

        // UNCOMMENT WHEN THE REAL MODEL IS READY /////////////////
        // landmarks of interest
        // int landmarksOfInterest[] = {55,87,57,35,144,129,39,1,95,113,45,118,149,
        //                              130,123,41,131,43,105,151,34,117,52,51,145,
        //                              125,62,59,109,152,27,11,22,69,146,103,47,
        //                              18,44,119};

        // for (int i = 0; i < 40; i++) {
        //     // since landmarks of interest are coming from Weka, subtract 1 from each
        //     std::cout << currentLandmarks[landmarksOfInterest[i]-1] -
        //                  neutralLandmarks[landmarksOfInterest[i]-1] << ',';
        // }
        ///////////////////////////////////////////////////////////

                // for (int i = 0; i < 40; i++) {
        //     // since landmarks of interest are coming from Weka, subtract 1 from each
        //     std::cout << currentLandmarks[landmarksOfInterest[i]-1] -
        //                  neutralLandmarks[landmarksOfInterest[i]-1] << ',';
        // }

        // double sample[] = {0.3456,0.501736,0.637789,0.434131,0.67091,0.506209,0.774421,0.57365,0.801669,0.369103,0.63683,0.427438,0.557188,0,0.489473,0.330205,0.49362,0.432366,0.478647,0.625093,0.610607,0.502084,0.498532,0.442381,0.599881,0.468418,0.523966,0.774152,0.542852,0,0.532809,0.782858,0.35473,0.338028,0.383349,0.25971,0.339092,0.35287,0.290431,0.32713,0.33519,0.365277,0.358864,0.429843,0.297704,0.372112,0.309986,0.331853,0.364021,0.22099,0.460859,0.329719,0.391221,0.30281,0.314147,0.334992,0.402591,0.555468,0.393714,0.531956,0.383657,0.611547,0.401101,0.644529,0.399358,0.686666,0.396547,0.632823,0.340352,0.558175,0.360233,0.603732,0.37594,0.648469,0.378973,0.64862,0.460551,0.632953,0.501103,0.632953,0.362779,0.655531,0.396638,0.663212,0.413725,0.660684,0.461478,0.656263,0.443778,0.57905,0.472049,0.592845,0.430782,0.619845,0.395356,0.652497,0.363975,0.547149,0.333895,0.556984,0.342175,0.579259,0.32885,0.623029,0.383779,0.631062,0.516293,0.5802,0.500714,0.430809,0.528475,0.508128,0.452684,0.605384,0.250261,0.564742,0.353824,0.436609,0.380734,0.570744,0.392296,0.583438,0.419357,0.605971,0.446168,0.649736,0.495483,0.586518,0.468518,0.563903,0.532263,0.527867,0.5158,0.638589,0.514193,0.682893,0.438192,0.682932,0.42495,0.671373,0.517377,0.663978,0.500746,0.642687,0.555317,0.609606,0.59819,0.673121,0.595583,0.703065,0.407841,0.710612,0.34297,0.695015};
        // double sample[] = {0.709935,0.737423,0.046317,0.460824,0,0.41709,0.151818,0.347885,0.327972,0.450519,0.50971,0.574647,0.532788,0,0.534658,0.451644,0.945861,0.481532,1,0.320895,1,0.379883,1,0.503664,0.314696,0.853222,0.884213,0.986928,0.799304,0,0.722609,0.798075,0.63274,1,0.685982,1,0.675579,0.92109,0.597837,0.960307,0.598118,0.954382,0.654399,0.922953,0.752669,0.905074,0.67379,1,0.589034,0.992844,0.589718,0.888503,0.636693,0.935754,0.666611,0.930642,0.934832,0.947407,0.652876,0.933375,0.617659,0.777678,0.602003,0.810578,0.653374,0.794802,0.649094,0.803351,0.659965,0.725723,0.643821,0.671804,0.607885,0.701745,0.585981,0.708935,0.656481,0.779908,1,0.779908,0.91895,0.785111,0.944503,0.836344,0.904962,0.865151,0.892942,0.849159,0.820447,0.849849,0.879869,0.733766,0.906933,0.711132,0.941298,0.720094,0.609799,0.551227,0.646107,0.554804,0.652654,0.568851,0.603585,0.416761,0.627371,0.446613,0.66425,0.401981,0.503661,0.236905,0.62065,0.387325,0.623608,0.3981,0.55101,0.390597,0.729705,0.157411,0.502458,0.415805,0.520895,0.382803,0.589867,0.390017,0.609469,0.421812,0.64427,0.393198,0.576059,0.367406,0.516002,0.424324,0.586127,0.370762,0.606329,0.395442,0.510969,0.39212,0.49242,0.495793,0.62481,0.462601,0.562901,0.473407,0.507692,0.390611,0.60376,0.433222,0.626444,0.466274,0.508405,0.47407,0.473301,0.477221};

        // Attributes selected with WEKA's SVM attribute filtering algorithm
        // int selectedLandmarks[] = {123,95,57,90,118,39,125,61,109,50,144,93,130,151,42,41,65,131,71,51,3,62,113,129,145,54,128,59,47,152,149,127,141,55,15,119,43,10,117,17};
        for (int i = 0; i < kNumAttr; ++i) {
            // printf("%f\n", currentLandmarks[i]-neutralLandmarks[i]);
            detection_features[i].index = i+1;
            detection_features[i].value = landmarks[i];
            // printf("%f\n", detection_features[i].value);
            // detection_features[i].value = (currentLandmarks[selectedLandmarks[i]] -
            //                               neutralLandmarks[selectedLandmarks[i]] + 0.08)*7.0;
            // detection_features[i].value = sample[i];
        }
        detection_features[kNumAttr].index = -1;

        int predict_label = svm_predict_probability(model, detection_features, prob_estimates);
        // for (int i = 0; i < kNumClasses; ++i) {
        //     // if (labels[i] == HAPPY) {
        //     //     printf("%f\n", prob_estimates[i]);
        //     // }
        //     switch(labels[i]) {
        //     case NEUTRAL:
        //         printf("NEUTRAL: %f\n", prob_estimates[i]);
        //         break;
        //     case ANGRY:
        //         printf("ANGRY: %f\n", prob_estimates[i]);
        //         break;
        //     case DISGUST:
        //         printf("DISGUST: %f\n", prob_estimates[i]);
        //         break;
        //     case FEAR:
        //         printf("FEAR: %f\n", prob_estimates[i]);
        //         break;
        //     case HAPPY:
        //         printf("HAPPY: %f\n", prob_estimates[i]);
        //         break;
        //     case SURPRISE:
        //         printf("SURPRISE: %f\n", prob_estimates[i]);
        //         break;
        //     default:
        //         printf("NOT FOUND\n");
        // }
        // }
        switch(predict_label) {
            case NEUTRAL:
                printf("NEUTRAL\n");
                break;
            case ANGRY:
                printf("ANGRY\n");
                break;
            case CONTEMPT:
                printf("CONTEMPT\n");
                break;
            case DISGUST:
                printf("DISGUST\n");
                break;
            case FEAR:
                printf("FEAR\n");
                break;
            case HAPPY:
                printf("HAPPY\n");
                break;
            case SAD:
                printf("SAD\n");
                break;
            case SURPRISE:
                printf("SURPRISE\n");
                break;
            default:
                printf("NOT FOUND\n");
        }
        putText(frame, "Neutral Face", Point(10,20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);
        putText(frame, "Happy Face", Point(10,35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);
        putText(frame, "Surprise Face", Point(10,50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);
        putText(frame, "Disgust Face", Point(10,65), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);
        putText(frame, "Angry Face", Point(10,80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);
        putText(frame, "Fear Face", Point(10,95), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);

        for (int i = 0; i < kNumClasses; ++i) {
            if (labels[i] == NEUTRAL) {
                line( frame, Point( 150, 15 ), Point( 150+300*prob_estimates[i], 15), Scalar( 0,0,255 ),  10, 8 );
               
            }
            if (labels[i] == HAPPY) {
                line( frame, Point( 150, 30 ), Point( 150+300*prob_estimates[i], 30), Scalar( 0,0,255 ),  10, 8 );
                
            }
             if (labels[i] == SURPRISE) {
                line( frame, Point( 150, 45 ), Point( 150+300*prob_estimates[i], 45), Scalar( 0,0,255 ),  10, 8 );
            }
            if (labels[i] == DISGUST) {
                line( frame, Point( 150, 60 ), Point( 150+300*prob_estimates[i], 60), Scalar( 0,0,255 ),  10, 8 );
                
            }
            if (labels[i] == ANGRY) {
                line( frame, Point( 150, 75 ), Point( 150+300*prob_estimates[i], 75), Scalar( 0,0,255 ),  10, 8 );
                
            }
             if (labels[i] == FEAR) {
                line( frame, Point( 150, 90 ), Point( 150+300*prob_estimates[i], 90), Scalar( 0,0,255 ),  10, 8 );
                
            }
        }
       
        cv::imshow("input", frame);
        // cv::imshow("plot", plotFrame);
        if(cv::waitKey(30) >= 0) break;
    }
    svm_free_and_destroy_model(&model);

    return 0;
}
