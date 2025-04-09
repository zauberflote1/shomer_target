#include "ShomerCore.hpp"

//TODO: MT FOR BUFFER OF IMAGES TO PROCESS

ShomerCore::~ShomerCore() {
       if (process_thread_.joinable()) {
            process_thread_.join();
        }
}

void ShomerCore::feed_image(ShomerImage image_) {
    //PREPROCESS IMAGE
    cv::Mat imageMono;
    //CONVERT IF NEEDED
    if (image_.image.channels() != 1) {
        if (options.image_type == ImageType::BGR) {
            cv::cvtColor(image_.image, imageMono, cv::COLOR_BGR2GRAY);
            //CONVERT ORIGINAL TO HSV
            cv::cvtColor(image_.image, image_.image, cv::COLOR_BGR2HSV);
        } else if (options.image_type == ImageType::YUV) {
            cv::cvtColor(image_.image, image_.image,cv::COLOR_YUV2BGR_Y422);
            cv::cvtColor(image_.image, imageMono, cv::COLOR_BGR2GRAY);
            //CONVERT ORIGINAL TO HSV
            cv::cvtColor(image_.image, image_.image, cv::COLOR_BGR2HSV);
        }
    } else { 
        imageMono = image_.image.clone();
    }
    preprocessedImage = preprocessImage(imageMono);
    std::optional<std::vector<BlobShomer>> blobShomerVec;
    //FIND CONTOURS
    if (options.mono) {
        blobShomerVec = findAndCalcContoursMono(preprocessedImage);
    } else {
        blobShomerVec = findAndCalcContours(preprocessedImage, image_.image);
    }
    if (blobShomerVec) {
        std::vector<BlobShomer> best_blobs;
        resultBlobs.clear();
        if (options.mono) {
            best_blobs = selectBlobsMono(blobShomerVec.value(), options.blob_config.min_circularity);
        } else {
            best_blobs = selectBlobs(blobShomerVec.value(), options.blob_config.min_circularity);
        }
        if (!best_blobs.empty()) {
            resultBlobs = best_blobs;
            std::vector<Blob> blobs;
            blobs.reserve(best_blobs.size());
            for (const auto& blobShomer : best_blobs) {
                blobs.emplace_back(blobShomer.blob);
            }
            processBlobs(blobs, image_.timestamp);
        }
    } else {
        printf("No valid contours found.\n");
    }
}

cv::Mat ShomerCore::preprocessImage(const cv::Mat& image) {
        cv::Mat blurred, thresholded;

        cv::GaussianBlur(image, blurred, cv::Size(options.kernel_size_gaussian, options.kernel_size_gaussian), 0);
        double threshValue = options.image_threshold;
        cv::threshold(blurred, thresholded, threshValue, 255, cv::THRESH_BINARY);
        // cv::GaussianBlur(thresholded, thresholded, cv::Size(options.kernel_size_gaussian, options.kernel_size_gaussian), 1);
        cv::Mat morph_kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(options.kernel_size_morph, options.kernel_size_morph));
        cv::morphologyEx(thresholded, thresholded, cv::MORPH_CLOSE, morph_kernel, cv::Point(-1, -1), options.it_morph_close);
        if (options.dilate) {
            cv::morphologyEx(thresholded, thresholded, cv::MORPH_DILATE, morph_kernel, cv::Point(-1, -1), options.it_morph_dilate);
        }
        return thresholded;
}

std::optional<std::vector<BlobShomer>> ShomerCore::findAndCalcContours(const cv::Mat &image, const cv::Mat &originalImageHSV) {
        std::vector<std::vector<cv::Point>> contours;

        auto start = std::chrono::high_resolution_clock::now(); //TIME MEASUREMENT COUNTOURS START
        //FIND CONTOURS IN THE IMAGE
        cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        auto end = std::chrono::high_resolution_clock::now(); //TIME MEASUREMENT COUNTOURS END
        std::chrono::duration<double> duration = end - start;
        printf("Time to find contours: %f seconds\n", duration.count());

        //CHECK IF CONTOURS ARE VALID
        if (contours.empty() || contours.size() > 1000) {
            printf("ERROR: %lu contours found\n", contours.size());
            return std::nullopt;
        }

        //PREPARE FOR BLOB PROPERTIES CALCULATION
        std::vector<BlobShomer> blobs;
        blobs.reserve(contours.size());

        start = std::chrono::high_resolution_clock::now(); //TIME MEASUREMENT BLOBS START

        //PARALLELIZE BLOB CALCULATIONS
        std::vector<std::future<void>> futures;
        std::mutex blobs_mutex;
        int num_threads_used = std::min(contours.size(), static_cast<size_t>(options.num_threads));
        int contours_per_thread = contours.size() / num_threads_used;
        int remainder = contours.size() % num_threads_used;
        int start_idx = 0;
        for (int t = 0; t < num_threads_used; ++t) {
            int end_idx = start_idx + contours_per_thread + (t < remainder ? 1 : 0);
            futures.emplace_back(std::async(std::launch::async, [this, t, start_idx, end_idx, contours_per_thread, remainder, &contours, &blobs, &blobs_mutex, &originalImageHSV]() {

                for (int i = start_idx; i < end_idx; ++i) {
                    const auto &contour = contours[i];

                    //CHECK AREA OF CONTOUR (m00)
                    cv::Moments moments = cv::moments(contour);
                    if (moments.m00 < options.blob_config.min_area|| moments.m00 > options.blob_config.max_area) { //30 2500
                        continue;
                    }
                    //CALCULATE BLOB PROPERTIES
                    double perimeter = cv::arcLength(contour, true);
                    double circularity = (4 * CV_PI * moments.m00) / (perimeter * perimeter);
                    double x = moments.m10 / moments.m00; //CENTROID X
                    double y = moments.m01 / moments.m00; //CENTROID Y
                    
                    //HUE EXTRACTION PROCESS
                    //BOUND THE CONTOUR
                    cv::Rect boundingRect = cv::boundingRect(contour);
                    cv::Mat BlobRegion = originalImageHSV(boundingRect);
                    //DECLARE HSV MASK TO FILTER SATURATION
                    cv::Mat maskHSV;
                    //FILTER OUT BAD PIXELS
                    cv::inRange(BlobRegion, cv::Scalar(0, options.blob_config.saturation_threshold, 0), cv::Scalar(180, 255, 255), maskHSV);
                    //CALCULATE HUE --> meanHSV[0] 0-180
                    cv::Scalar meanHSV;
                    if (!options.blob_config.circular_mean_hue) {
                        meanHSV = cv::mean(BlobRegion, maskHSV);
                        if (meanHSV[0] < options.blob_config.lb_hue || meanHSV[0] > options.blob_config.ub_hue) {
                            continue;
                        }
                    } else { //CIRCULAR MEAN ALGORITHM
                        std::vector<cv::Mat> hsvChannels;
                        cv::split(BlobRegion, hsvChannels);
                        cv::Mat hueChannel = hsvChannels[0];

                        double sumSin = 0.0;
                        double sumCos = 0.0;
                        int count = 0;

                        for (int i = 0; i < hueChannel.rows; ++i) {
                            for (int j = 0; j < hueChannel.cols; ++j) {
                                if (maskHSV.at<uchar>(i, j) != 0) {  
                                double hue = hueChannel.at<uchar>(i, j) * 2.0 * CV_PI / 180.0;
                                sumSin += std::sin(hue);
                                sumCos += std::cos(hue);
                                ++count;
                                }
                            }
                        }  

                        double meanAngle = std::atan2(sumSin / count, sumCos / count) * 180.0 / CV_PI;
                        if (meanAngle < 0) meanAngle += 360.0;
                            meanHSV[0] = meanAngle/ 2.0;

                        
                        //CHECK HUE CONDITIONS PER BLOB COLOR
                        if (std::isnan(meanHSV[0]) || std::isinf(meanHSV[0])) {
                            continue;
                        }
                        if (meanHSV[0] < 20){ //WRAP VALUE ADJUST AS NEEDED
                            meanHSV[0] = 180 - meanHSV[0];
                        }
                        if (meanHSV[0] < options.blob_config.lb_hue || meanHSV[0] > options.blob_config.ub_hue) {
                            continue;
                        }
                    }


                    BlobShomer blobShomer_;
                    blobShomer_.blob = {x, y};
                    blobShomer_.properties = {perimeter, moments.m00, circularity, meanHSV[0], boundingRect};

                    std::lock_guard<std::mutex> lock(blobs_mutex);
                    blobs.emplace_back(std::move(blobShomer_));
                }
            }));
            start_idx = end_idx;
        }

        for (auto &fut : futures) {
            fut.get();
        }

        end = std::chrono::high_resolution_clock::now(); //TIME MEASUREMENT BLOBS END
        duration = end - start;
        printf("Time to calculate blobs: %f seconds\n", duration.count());
        return blobs;
}

std::optional<std::vector<BlobShomer>> ShomerCore::findAndCalcContoursMono(const cv::Mat &image) {
        std::vector<std::vector<cv::Point>> contours;

        auto start = std::chrono::high_resolution_clock::now(); //TIME MEASUREMENT COUNTOURS START
        //FIND CONTOURS IN THE IMAGE
        cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        auto end = std::chrono::high_resolution_clock::now(); //TIME MEASUREMENT COUNTOURS END
        std::chrono::duration<double> duration = end - start;
        printf("Time to find contours: %f seconds\n", duration.count());

        //CHECK IF CONTOURS ARE VALID
        if (contours.empty() || contours.size() > 1000) {
            printf("ERROR: %lu contours found\n", contours.size());
            return std::nullopt;
        }

        //PREPARE FOR BLOB PROPERTIES CALCULATION
        std::vector<BlobShomer> blobs;
        blobs.reserve(contours.size());

        start = std::chrono::high_resolution_clock::now(); //TIME MEASUREMENT BLOBS START

        //PARALLELIZE BLOB CALCULATIONS
        std::vector<std::future<void>> futures;
        std::mutex blobs_mutex;
              int num_threads_used = std::min(contours.size(), static_cast<size_t>(options.num_threads));
        int contours_per_thread = contours.size() / num_threads_used;
        int remainder = contours.size() % num_threads_used;
        int start_idx = 0;
        for (int t = 0; t < num_threads_used; ++t) {
            int end_idx = start_idx + contours_per_thread + (t < remainder ? 1 : 0);
            futures.emplace_back(std::async(std::launch::async, [this, t, start_idx, end_idx, contours_per_thread, remainder, &contours, &blobs, &blobs_mutex]() {

                for (int i = start_idx; i < end_idx; ++i) {
                    const auto &contour = contours[i];

                    //CHECK AREA OF CONTOUR (m00)
                    cv::Moments moments = cv::moments(contour);
                    if (moments.m00 < options.blob_config.min_area|| moments.m00 > options.blob_config.max_area) { //30 2500
                        continue;
                    }
                    //CALCULATE BLOB PROPERTIES
                    double perimeter = cv::arcLength(contour, true);
                    double circularity = (4 * CV_PI * moments.m00) / (perimeter * perimeter);
                    double x = moments.m10 / moments.m00; //CENTROID X
                    double y = moments.m01 / moments.m00; //CENTROID Y
                    
                    cv::Rect boundingRect = cv::boundingRect(contour);


                    BlobShomer blobShomer_;
                    blobShomer_.blob = {x, y};
                    blobShomer_.properties = {perimeter, moments.m00, circularity, 180, boundingRect};

                    std::lock_guard<std::mutex> lock(blobs_mutex);
                    blobs.emplace_back(std::move(blobShomer_));
                }
            }));
            start_idx = end_idx;
        }

        for (auto &fut : futures) {
            fut.get();
        }

        end = std::chrono::high_resolution_clock::now(); //TIME MEASUREMENT BLOBS END
        duration = end - start;
        printf("Time to calculate blobs: %f seconds\n", duration.count());
        return blobs;
}

std::vector<BlobShomer> ShomerCore::selectBlobs(const std::vector<BlobShomer>& blobs, double min_circularity) {  //TODO KD-TREE
        std::vector<BlobShomer> filtered_blobs;
        filtered_blobs.reserve(blobs.size());

        if (blobs.size() < 4) {
            printf("Not enough blobs < 4.\n");
            return {};
        }

        for (const auto& blob : blobs) {
            if (blob.properties.circularity >= min_circularity) {
                filtered_blobs.emplace_back(blob);
            }
        }

        if (filtered_blobs.size() < 4) {
            printf("Not enough blobs with required circularity.\n");
            return {};
        }

        double min_variation = std::numeric_limits<double>::max();
        std::vector<BlobShomer> best_group;

        for (size_t i = 0; i < filtered_blobs.size() - 3; ++i) {
            for (size_t j = i + 1; j < filtered_blobs.size() - 2; ++j) {
                for (size_t k = j + 1; k < filtered_blobs.size() - 1; ++k) {
                    for (size_t l = k + 1; l < filtered_blobs.size(); ++l) {

                        
                        double dist_ij = std::hypot(filtered_blobs[i].blob.x - filtered_blobs[j].blob.x, filtered_blobs[i].blob.y - filtered_blobs[j].blob.y);
                        double dist_ik = std::hypot(filtered_blobs[i].blob.x - filtered_blobs[k].blob.x, filtered_blobs[i].blob.y - filtered_blobs[k].blob.y);
                        double dist_il = std::hypot(filtered_blobs[i].blob.x - filtered_blobs[l].blob.x, filtered_blobs[i].blob.y - filtered_blobs[l].blob.y);
                        double dist_jk = std::hypot(filtered_blobs[j].blob.x - filtered_blobs[k].blob.x, filtered_blobs[j].blob.y - filtered_blobs[k].blob.y);
                        double dist_jl = std::hypot(filtered_blobs[j].blob.x - filtered_blobs[l].blob.x, filtered_blobs[j].blob.y - filtered_blobs[l].blob.y);
                        double dist_kl = std::hypot(filtered_blobs[k].blob.x - filtered_blobs[l].blob.x, filtered_blobs[k].blob.y - filtered_blobs[l].blob.y);

                        std::vector<double> distances = {dist_ij, dist_ik, dist_il, dist_jk, dist_jl, dist_kl};
                        double max_distance = *std::max_element(distances.begin(), distances.end());
                        double mean_distance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
                        double distance_variance = std::accumulate(distances.begin(), distances.end(), 0.0,
                            [mean_distance](double sum, double distance) {
                                return sum + std::pow(distance - mean_distance, 2);
                            }) / distances.size();

                        
                        if (max_distance > options.max_distance_lim) {//500
                            continue;
                        }

                        
                        std::vector<double> areas = {
                            filtered_blobs[i].properties.m00,
                            filtered_blobs[j].properties.m00,
                            filtered_blobs[k].properties.m00,
                            filtered_blobs[l].properties.m00
                        };
                        double mean_area = std::accumulate(areas.begin(), areas.end(), 0.0) / areas.size();
                        double area_variance = std::accumulate(areas.begin(), areas.end(), 0.0,
                            [mean_area](double sum, double area) {
                                return sum + std::pow(area - mean_area, 2);
                            }) / areas.size();


                        std::vector<double> hues = {
                            filtered_blobs[i].properties.hue,
                            filtered_blobs[j].properties.hue,
                            filtered_blobs[k].properties.hue,
                            filtered_blobs[l].properties.hue
                        };
                        double mean_hue = std::accumulate(hues.begin(), hues.end(), 0.0) / areas.size();
                        double intensity_hues = std::accumulate(hues.begin(), hues.end(), 0.0,
                            [mean_hue](double sum, double hues) {
                                return sum + std::pow(hues - mean_hue, 2);
                            }) / hues.size();

                        double combined_variance = distance_variance + 1.5*area_variance; //+ intensity_hues;

                        if (combined_variance < min_variation) {
                            min_variation = combined_variance;
                            best_group = {filtered_blobs[i], filtered_blobs[j], filtered_blobs[k], filtered_blobs[l]};
                        }
                    }
                }
            }
        }

        return best_group;
}

    //SELECT BLOBS BASED ON CIRCULARITY AND VARIANCE --> NEEDS CLEANING AND OPTIMIZATION KD-TREE
std::vector<BlobShomer> ShomerCore::selectBlobsMono(const std::vector<BlobShomer>& blobs, double min_circularity) {
    std::vector<BlobShomer> filtered_blobs;
    filtered_blobs.reserve(blobs.size());

    if (blobs.size() < 4) {
        printf("Not enough blobs < 4.\n");
        return {};
    }

    for (const auto& blob : blobs) {
        if (blob.properties.circularity >= min_circularity) {
            filtered_blobs.emplace_back(blob);
        }
    }

    if (filtered_blobs.size() < 4) {
        printf("Not enough blobs with required circularity.\n");
        return {};
    }

    double min_variation = std::numeric_limits<double>::max();
    std::vector<BlobShomer> best_group;

    for (size_t i = 0; i < filtered_blobs.size() - 3; ++i) {
        for (size_t j = i + 1; j < filtered_blobs.size() - 2; ++j) {
            for (size_t k = j + 1; k < filtered_blobs.size() - 1; ++k) {
                for (size_t l = k + 1; l < filtered_blobs.size(); ++l) {

                    
                    double dist_ij = std::hypot(filtered_blobs[i].blob.x - filtered_blobs[j].blob.x, filtered_blobs[i].blob.y - filtered_blobs[j].blob.y);
                    double dist_ik = std::hypot(filtered_blobs[i].blob.x - filtered_blobs[k].blob.x, filtered_blobs[i].blob.y - filtered_blobs[k].blob.y);
                    double dist_il = std::hypot(filtered_blobs[i].blob.x - filtered_blobs[l].blob.x, filtered_blobs[i].blob.y - filtered_blobs[l].blob.y);
                    double dist_jk = std::hypot(filtered_blobs[j].blob.x - filtered_blobs[k].blob.x, filtered_blobs[j].blob.y - filtered_blobs[k].blob.y);
                    double dist_jl = std::hypot(filtered_blobs[j].blob.x - filtered_blobs[l].blob.x, filtered_blobs[j].blob.y - filtered_blobs[l].blob.y);
                    double dist_kl = std::hypot(filtered_blobs[k].blob.x - filtered_blobs[l].blob.x, filtered_blobs[k].blob.y - filtered_blobs[l].blob.y);

                    std::vector<double> distances = {dist_ij, dist_ik, dist_il, dist_jk, dist_jl, dist_kl};
                    double max_distance = *std::max_element(distances.begin(), distances.end());
                    double mean_distance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
                    double distance_variance = std::accumulate(distances.begin(), distances.end(), 0.0,
                        [mean_distance](double sum, double distance) {
                            return sum + std::pow(distance - mean_distance, 2);
                        }) / distances.size();

                    
                    if (max_distance > options.max_distance_lim) {
                        continue;
                    }

                    
                    std::vector<double> areas = {
                        filtered_blobs[i].properties.m00,
                        filtered_blobs[j].properties.m00,
                        filtered_blobs[k].properties.m00,
                        filtered_blobs[l].properties.m00
                    };
                    double mean_area = std::accumulate(areas.begin(), areas.end(), 0.0) / areas.size();
                    double area_variance = std::accumulate(areas.begin(), areas.end(), 0.0,
                        [mean_area](double sum, double area) {
                            return sum + std::pow(area - mean_area, 2);
                        }) / areas.size();


                    double combined_variance = distance_variance + 1.5*area_variance;

                    if (combined_variance < min_variation) {
                        min_variation = combined_variance;
                        best_group = {filtered_blobs[i], filtered_blobs[j], filtered_blobs[k], filtered_blobs[l]};
                    }
                }
            }
        }
    }

    return best_group;
}




// CameraPose fifo(const CameraPose& new_pose, const ros::Time& timestamp) {//TODO ROS CLEANUP
//         if (!pose_queue_.empty()) {
//             if (last_valid_timestamp - timestamp > ros::Duration(max_time_fifo)) { //TODO ROS CLEANUP
//                 pose_queue_.clear();
//             }
//             const CameraPose& last_pose = pose_queue_.back();

//             //eval translation diff
//             double translation_diff = (last_pose.t - new_pose.t).norm();
//             if (translation_diff > translation_threshold_) {
//                 printf("Current pose rotation is too different. Ignoring current pose.");
//                 reject_count++;
//                 if (reject_count > reject_limit) {
//                     pose_queue_.clear();
//                     reject_count = 0;
//                 }
//                 return last_pose;
//             }

//             //eval rotation diff using SLERP
//             Eigen::Quaterniond q_last(last_pose.R);
//             Eigen::Quaterniond q_new(new_pose.R);
//             double angle_diff = q_last.angularDistance(q_new);
//             if (angle_diff > rotation_threshold_) {
//                 printf("Current pose rotation is too different. Ignoring current pose.");
//                 reject_count++;
//                 if (reject_count > reject_limit) {
//                     pose_queue_.clear();
//                     reject_count = 0;
//                 }
//                 return last_pose;
//             }
//         }

//         pose_queue_.push_back(new_pose);
//         last_valid_timestamp = timestamp;
//         if (pose_queue_.size() > filter_size_) {
//             pose_queue_.pop_front();
//         }

//         //average translation
//         Eigen::Vector3d avg_t = Eigen::Vector3d::Zero();
//         for (const auto& pose : pose_queue_) {
//             avg_t += pose.t;
//         }
//         avg_t /= pose_queue_.size();

//         //average rotation using SLERP
//         Eigen::Quaterniond q_avg = Eigen::Quaterniond::Identity();
//         double weight = 1.0 / pose_queue_.size();
//         bool initialized = false;

//         for (const auto& pose : pose_queue_) {
//             Eigen::Quaterniond q_pose(pose.R);
//             if (!initialized) {
//                 q_avg = q_pose;
//                 initialized = true;
//             } else {
//                 q_avg = q_avg.slerp(weight, q_pose);
//             }
//         }

//         //quaternion back to rotation matrix
//         Eigen::Matrix3d avg_R = q_avg.normalized().toRotationMatrix();

//         CameraPose filtered_pose;
//         filtered_pose.R = avg_R;
//         filtered_pose.t = avg_t;
//         return filtered_pose;
//     }
void ShomerCore::processBlobs(const std::vector<Blob>& blobs, const int64_t& timestamp) {
    if (blobs.size() < 4) {
        printf("Not enough blobs to calculate 6DoF state.\n");
        return;
    }

    //GET UNDISTORTED AND SORTED MATRIX
    Eigen::Matrix<double, 4, 2> undistortedSortedPoints = helper.getUndistortedSortedPoints(blobs, options.fisheye);

   if (undistortedSortedPoints.isZero(0)) { 
        result.R.setZero();
        return;
   }
   result = engine.ShomerEngineSolver(undistortedSortedPoints);
}

std::optional<std::vector<BlobShomer>> ShomerCore::getBlobs() {
    if (resultBlobs.empty()) {
        return std::nullopt;
    }
    return resultBlobs;
}

std::optional<CameraPose> ShomerCore::getPose() {
   if (result.R.isZero(0) || resultBlobs.empty() ) {
       return std::nullopt;
   }
    return result;
}

std::optional<cv::Mat> ShomerCore::getPreprocessedImage() {
    if (preprocessedImage.empty()) {
        return std::nullopt;
    }
    return preprocessedImage;
}




