#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "yolo11.h"
#include "image_utils.h"
#include "image_drawing.h"
#include <string>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include "serial.hpp"
#include <pthread.h>
#include <sched.h>

#include <iomanip>
#include <sstream>

void saveFrameToFile(const cv::Mat& frame, const std::string& prefix = "frame") {
    // 获取当前时间戳作为文件名的一部分
    auto now = std::chrono::system_clock::now();
    auto time_in_seconds = std::chrono::system_clock::to_time_t(now);
    
    // 格式化时间戳为字符串
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_in_seconds), "%Y%m%d_%H%M%S");
    
    // 生成文件名（当前时间戳加上帧的编号或标签）
    std::string filename = prefix + "_" + ss.str() + ".jpg";
    
    // 保存帧为图片
    cv::imwrite(filename, frame);
    printf("Frame saved as: %s\n", filename.c_str());
}

using namespace std::chrono_literals;

int arr_put[6] = {0}; // 置物区数组
int arr_get[6] = {0}; // 取物区数组
std::mutex arr_mutex; // 数组访问互斥锁
bool put_complete = false; // 置物区处理完成标志

// --- 线程间通信 ---
std::mutex signal_mutex;
std::condition_variable signal_cv;
std::string current_signal;
bool new_signal = false;

// --- 功能函数 ---
// const char* coco_cls_to_name(int cls_id);

// 帧处理结果结构
struct FrameResult {
    std::vector<std::string> labels;
    int frame_count;
};

// 一帧推理
FrameResult processFrame(cv::Mat& frame, rknn_app_context_t* rknn_app_ctx) {
    FrameResult result;
    result.frame_count = 1;
    
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    src_image.width = frame.cols;
    src_image.height = frame.rows;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.size = frame.total() * frame.elemSize();
    src_image.virt_addr = (uint8_t*)malloc(src_image.size);
    memcpy(src_image.virt_addr, frame.data, src_image.size);

    object_detect_result_list od_results;
    int ret = inference_yolo11_model(rknn_app_ctx, &src_image, &od_results);
    
    if (ret == 0) {
        for (int i = 0; i < od_results.count; i++) {
            const char* label_name = coco_cls_to_name(od_results.results[i].cls_id);
            result.labels.push_back(label_name);
        }
    }

    saveFrameToFile(frame);

    free(src_image.virt_addr);
    return result;
}

// 多帧推理、统计结果
FrameResult processFrames(cv::VideoCapture& cap, rknn_app_context_t* rknn_app_ctx, int frame_count) {
    printf("get into processFrames\n");
    FrameResult aggregate_result;
    aggregate_result.frame_count = 0;
    
    for (int i = 0; i < frame_count; i++) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            continue;
            printf("empty frame----------\n");
        }
        // cv::imshow ("frames", frame);
        saveFrameToFile(frame, "frame_" + std::to_string(i));
        FrameResult frame_result = processFrame(frame, rknn_app_ctx);
        aggregate_result.frame_count += frame_result.frame_count;
        aggregate_result.labels.insert(aggregate_result.labels.end(), 
                                      frame_result.labels.begin(), 
                                      frame_result.labels.end());
    }
    printf("get out of processFrames\n");
    return aggregate_result;
}

// 数字打分
int getMostFrequentNumber(const FrameResult& result) {
    int counts[7] = {0}; // 对应0-6（0不使用）
    
    for (const auto& label : result.labels) {
        if (label == "1") counts[1]++;
        else if (label == "2") counts[2]++;
        else if (label == "3") counts[3]++;
        else if (label == "4") counts[4]++;
        else if (label == "5") counts[5]++;
        else if (label == "6") counts[6]++;
    }
    
    int max_count = 0;
    int frequent_number = 0;
    for (int i = 1; i <= 6; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            frequent_number = i;
        }
    }
    
    return frequent_number;
}

// 是否足够多的sharp
bool hasEnoughSharp(const FrameResult& result, int threshold) {
    int sharp_count = 0;
    for (const auto& label : result.labels) {
        if (label == "sharp") {
            sharp_count++;
        }
    }
    return sharp_count >= threshold;
}

// 置物区
void processPuttingOperations(cv::VideoCapture& cap, rknn_app_context_t* rknn_app_ctx) {
    std::unique_lock<std::mutex> lock(signal_mutex);
    //bool a11b_processed = false;

    while (true) {
        signal_cv.wait(lock, [&] { return new_signal; });
        new_signal = false;

        // 处理 a11b 命令
        //if (!a11b_processed && current_signal == "a11b") {
        if (current_signal == "a11b") {
            // 第一个10帧 -> 下标1
            FrameResult frames1 = processFrames(cap, rknn_app_ctx, 20);
            printf("frame a11b in\n");
            if (hasEnoughSharp(frames1, 5)) {
                arr_put[1] = getMostFrequentNumber(frames1);
                printf("arr_put[1] = %d\n", arr_put[1]);
            } else {
                arr_put[1] = 0;
                printf("sharp not enough [1]\n");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            
            // 第二个10帧 -> 下标2
            FrameResult frames2 = processFrames(cap, rknn_app_ctx, 20);
            if (hasEnoughSharp(frames2, 5)) {
                arr_put[2] = getMostFrequentNumber(frames2);
                printf("arr_put[2] = %d\n", arr_put[2]);
            } else {
                arr_put[2] = 0;
                printf("sharp not enough [2]\n");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            
            // 第三个10帧 -> 下标3
            FrameResult frames3 = processFrames(cap, rknn_app_ctx, 20);
            if (hasEnoughSharp(frames3, 5)) {
                arr_put[3] = getMostFrequentNumber(frames3);
                printf("arr_put[3] = %d\n", arr_put[3]);
            } else {
                arr_put[3] = 0;
                printf("sharp not enough [3]\n");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            
            // 第四个10帧 -> 下标4
            FrameResult frames4 = processFrames(cap, rknn_app_ctx, 20);
            if (hasEnoughSharp(frames4, 5)) {
                arr_put[4] = getMostFrequentNumber(frames4);
                printf("arr_put[4] = %d\n", arr_put[4]);
            } else {
                arr_put[4] = 0;
                printf("sharp not enough [4]\n");
            }
            
            //a11b_processed = true;
        }
        
        // 处理 a12b 命令 -> 下标0
        else if (current_signal == "a12b") {
            printf("get message a12b\n");
            FrameResult frames = processFrames(cap, rknn_app_ctx, 20);
            printf("frame a12b in\n");
            if (hasEnoughSharp(frames, 5)) {
                arr_put[0] = getMostFrequentNumber(frames);
                printf("arr_put[0] = %d\n", arr_put[0]);
            } else {
                arr_put[0] = 0;
                printf("sharp not enough [0]\n");
            }
        }
        
        // 处理 a13b 命令 -> 下标5
        else if (current_signal == "a13b") {
            FrameResult frames = processFrames(cap, rknn_app_ctx, 20);
            printf("frame a13b in\n");
            if (hasEnoughSharp(frames, 5)) {
                arr_put[5] = getMostFrequentNumber(frames);
                printf("arr_put[5] = %d\n", arr_put[5]);
            } else {
                arr_put[5] = 0;
                printf("sharp not enough [5]\n");
            }
            
            // 所有置物操作完成
            std::lock_guard<std::mutex> lock(arr_mutex);
            //put_complete = true;
            printf("Putting operations completed.\n");
            for (int i = 0; i != 6; i++) {
                printf("arr_put [%d] = %d", i, arr_put[i]);
            }
            break;
        }
    }
}
char mapNumberToChar(int num) {
    switch(num) {
        case 1: return 'a';
        case 2: return 'b';
        case 3: return 'c';
        case 4: return 'd';
        case 5: return 'e';
        case 6: return 'f';
        case 9: return '9';
        default: return '0';
    }
}
// 取物区处理
void processGettingOperations(cv::VideoCapture& cap, rknn_app_context_t* rknn_app_ctx) {
    // printf("get into processGettingOperations------------------\n");
    int send_index = 0;
    // 等待置物区
    // while (!put_complete) {
    //     std::this_thread::sleep_for(100ms);
    // }

    SerialPort serial("/dev/ttyUSB0", 115200);
    int get_index = 0;  // arr_get填充索引
    send_index = 0;

    while (true) {
        std::unique_lock<std::mutex> lock(signal_mutex);
        signal_cv.wait(lock, [&]{ return new_signal; });
        new_signal = false;
        
        if (current_signal == "a21b" && get_index < 6) {
            // 处理10帧获取高频数字
            FrameResult frames = processFrames(cap, rknn_app_ctx, 20);
            int num = getMostFrequentNumber(frames);
            ////////////////////////////////////////////// for test
            // static int flag = 0;
            // if (flag == 0) {
            //     char frame[5] = {'A', 'A', 'a', 'E', '\0'};
            //     serial.write(frame);
            //     printf("send_message: %s\n", frame);
            //     flag ++;
            // } else if(flag == 1) {
            //     char frame[5] = {'A', 'A', 'f', 'E', '\0'};
            //     serial.write(frame);
            //     printf("send_message: %s\n", frame);
            //     // flag ++;
            // }
            ////////////////////////////////////////////

            
            // 在arr_put中查找匹配项
            bool found = false;
            {
                std::lock_guard<std::mutex> arr_lock(arr_mutex);
                for (int i = 0; i < 6; i++) {
                    if (arr_put[i] == num) {
                        arr_get[get_index] = i;
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    arr_get[get_index] = 9; // 未找到标记
                }
                get_index++;
            }
        }

        // a31b命令处理
        else if (current_signal == "a31b") {
            if (send_index < 6) {
                int value = arr_get[send_index];
                char data_char = mapNumberToChar(value);
                
                构建帧：A + 数据 + E
                char frame[4] = {'A', data_char, 'E', '\0'};
                serial.write(frame);
                //////////////////// for test
                // char frame[5] = {'A', 'B', 'A', 'E', '\0'};
                // serial.write(frame);
                // printf("send_message: %s\n", frame);
                /////////////////////
                send_index++;
            } else {
                // 错误帧：A + X + E
                char error_frame[4] = {'A', 'X', 'E', '\0'};
                serial.write(error_frame);
            }
        }

        // a30b命令处理
        else if (current_signal == "a30b") {
            int found_value = 9;
            int found_index = -1;

            // 向前搜索非9元素
            for (int i = send_index; i >= 0; i--) {
                if (arr_get[i] != 9) {
                    found_value = arr_get[i];
                    found_index = i;
                    break;
                }
            }
            
            // 映射数据
            char data_char = mapNumberToChar(found_value);
            
            // 构建帧：A + 数据 + E
            char frame[4] = {'A', data_char, 'E', '\0'};
            serial.write(frame);
            
            // 更新发送指针
            if (found_index != -1) {
                send_index = found_index + 1;
            } else {
                send_index++;
            }
        }
    }
}

void processCamera(rknn_app_context_t* rknn_app_ctx, int camera_id, const char* camera_path) {
    printf("Opening camera %d at %s\n", camera_id, camera_path);
    
    cv::VideoCapture cap(camera_path);
    if (!cap.isOpened()) {
        printf("Error: Cannot open camera %d\n", camera_id);
        return;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    if (camera_id == 0) {
        processGettingOperations(cap, rknn_app_ctx);
    } else {
        processPuttingOperations(cap, rknn_app_ctx);
    }
    cap.release();
}

void serialListener() {
    try {
        SerialPort serial("/dev/ttyUSB0", 115200);
        
        while (true) {
            // 命令（...b
            std::string command = serial.readUntil();
            std::cout << "Received command: " << command << std::endl;
            
            // 处理命令
            {
                std::lock_guard<std::mutex> lock(signal_mutex);
                current_signal = command;
                new_signal = true;
            }
            signal_cv.notify_all();
            
        }
    } catch (const std::exception& e) {
        std::cerr << "Serial error: " << e.what() << std::endl;
    }
}

// 设置线程亲和性函数实现
void setThreadAffinity(pthread_t thread, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    
    int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error setting thread affinity: " << rc << std::endl;
    }
}

int main(int argc, char **argv) {
    printf("Initializing YOLOv11 models...\n");
    const char* model_path = "/root/rk3588_detect/rknn_model/best.rknn";
    
    // 取物区init
    rknn_app_context_t rknn_app_ctx_0;
    memset(&rknn_app_ctx_0, 0, sizeof(rknn_app_context_t));
    if (init_yolo11_model(model_path, &rknn_app_ctx_0) != 0) {
        printf("Model initialization failed!\n");
        return -1;
    }
    
    // 置物区init
    rknn_app_context_t rknn_app_ctx_1;
    memset(&rknn_app_ctx_1, 0, sizeof(rknn_app_context_t));
    if (init_yolo11_model(model_path, &rknn_app_ctx_1) != 0) {
        printf("Model initialization failed!\n");
        return -1;
    }
    
    std::thread serial_thread(serialListener);
    setThreadAffinity(serial_thread.native_handle(), 7);

    std::thread put_camera_thread(processCamera, &rknn_app_ctx_1, 2, "/dev/camera_1080");
    setThreadAffinity(put_camera_thread.native_handle(), 6);

    std::thread get_camera_thread(processCamera, &rknn_app_ctx_0, 0, "/dev/camera_720");
    setThreadAffinity(get_camera_thread.native_handle(), 5);
    
    serial_thread.join();
    put_camera_thread.join();
    get_camera_thread.join();
    
    release_yolo11_model(&rknn_app_ctx_0);
    release_yolo11_model(&rknn_app_ctx_1);
    printf("Exit.\n");
    
    return 0;
}
