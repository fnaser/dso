//
// Created by fnaser on 10/10/18.
//

#include "IOWrapper/OutputWrapper/RegistrationOutputWrapper.h"

dso::IOWrap::RegistrationOutputWrapper::RegistrationOutputWrapper(int w, int h, bool nogui)
{
    printf("OUT: Created SampleOutputWrapper\n");

    //TODO read vec from txt

    //TODO params for seq 28
    this->world_pts_.push_back(
            Eigen::Vector3d(-1.5977, -2.8942, 4.7679));
    this->world_pts_.push_back(
            Eigen::Vector3d(-3.2313, -2.9793, 3.1786));
    this->world_pts_.push_back(
            Eigen::Vector3d(0.3451, 0.4708, 1.1789));
    assert(this->world_pts_.size() == 3); //TODO param

    //TODO params for seq 28
    this->roi_pts_.push_back(
            Eigen::Vector3d(-0.5596, -0.0249, 1.0518)); // top right
    this->roi_pts_.push_back(
            Eigen::Vector3d(-0.6352, -0.0863, 1.0748)); // top left
    this->roi_pts_.push_back(
            Eigen::Vector3d(-0.6801, -0.0633, 0.9885)); // bottom left
    this->roi_pts_.push_back(
            Eigen::Vector3d(-0.6046, -0.0019, 0.9655)); // bottom right
    assert(roi_pts_.size() >= 1); //TODO param

    this->roi_pts_rectified_.push_back(
            cv::Point(100,0));
    this->roi_pts_rectified_.push_back(
            cv::Point(0,0));
    this->roi_pts_rectified_.push_back(
            cv::Point(0,100));
    this->roi_pts_rectified_.push_back(
            cv::Point(100,100)); //TODO param //TODO needed?
    assert(roi_pts_rectified_.size() == roi_pts_.size());

    this->w_ = w;
    this->h_ = h;
    this->nogui_ = nogui;
    this->seq_length_ = 10; //TODO param
    this->start_idx_ = -1;
    this->seq_idx_ = 0;

    this->label_ = 0; //0: static 1: dynamic //TODO param
    this->store_imgs_ = true; //TODO param
    this->img_folder_ = "/home/fnaser/Pictures/testing_dso_H/cropped_"; //TODO param
    this->csv_point_cloud_ = "/home/fnaser/example.csv"; //TODO param
    this->csv_seq_labels_ = "/home/fnaser/labels_seq_28.csv"; //TODO param

    checkFunctionOutput();
}

dso::IOWrap::RegistrationOutputWrapper::~RegistrationOutputWrapper()
{
    printf("OUT: Destroyed SampleOutputWrapper\n");
    this->vectorToFile();
    this->labelsToFile();
}

void dso::IOWrap::RegistrationOutputWrapper::publishCamPose(dso::FrameShell* frame,
                                                            dso::CalibHessian* HCalib) {
    K_ = Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3>::Zero(3, 3);
    K_(0,0) = HCalib->fxl();
    K_(1,1) = HCalib->fyl();
    K_(0,2) = HCalib->cxl();
    K_(1,2) = HCalib->cyl();
    K_(2,2) = 1;

    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M =
            Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4>::Zero(4, 4);
    M.block<3,4>(0,0) = frame->camToWorld.matrix3x4();
    M(3,3) = 1.0;
    M = M.inverse().eval();
    std::pair<int, Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4>> tmp(frame->id, M);
    seq_Ms_.insert(tmp);

    setStartIdx(frame->id);

    if (seq_Ms_.find(start_idx_) != seq_Ms_.end() &&
        seq_Ms_.find(start_idx_+seq_length_) != seq_Ms_.end() &&
        seq_imgs_.find(start_idx_) != seq_imgs_.end() &&
        seq_imgs_.find(start_idx_+seq_length_) != seq_imgs_.end()) {

        this->showImgs();

        seq_Ms_.erase(seq_Ms_.begin(), seq_Ms_.find(start_idx_));
        seq_imgs_.erase(seq_imgs_.begin(), seq_imgs_.find(start_idx_));
        start_idx_ = -1;
        seq_idx_++;
    }

    if (frame->id % 100 == 0) { //TODO param
        this->vectorToFile();
        this->labelsToFile();
    }
}

void dso::IOWrap::RegistrationOutputWrapper::pushLiveFrame(FrameHessian* image) {
    MinimalImageB3* internalVideoImg = new MinimalImageB3(w_,h_);
    internalVideoImg->setBlack();

    for(int i=0;i<w_*h_;i++) {
        internalVideoImg->data[i][0] =
        internalVideoImg->data[i][1] =
        internalVideoImg->data[i][2] =
                image->dI[i][0] * 0.8 > 255.0f ? 255.0 : image->dI[i][0] * 0.8;
    }

    int frameID = image->shell->id;
    cv::Mat tmp = cv::Mat(h_, w_, CV_8UC3, internalVideoImg->data);
    seq_imgs_.insert(std::pair<int, cv::Mat>(frameID, tmp));

    setStartIdx(frameID);
}

void dso::IOWrap::RegistrationOutputWrapper::setStartIdx(int frameID) {
    if (start_idx_ == -1 || seq_Ms_.empty()) {
        start_idx_ = frameID;
    }
}

void dso::IOWrap::RegistrationOutputWrapper::publishKeyframes(
        std::vector<FrameHessian*> &frames,
        bool final, CalibHessian* HCalib) {

    float fx = HCalib->fxl();
    float fy = HCalib->fyl();
    float cx = HCalib->cxl();
    float cy = HCalib->cyl();
    float fxi = 1/fx;
    float fyi = 1/fy;
    float cxi = -cx / fx;
    float cyi = -cy / fy;

    for(int i=0; i < frames.size() && i < 1; i++) { //TODO param

        const Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 4> mTwc = frames[i]->shell->camToWorld.matrix3x4();

        for (PointHessian *p : frames[i]->pointHessians) {

            float depth = 1.0f / p->idepth_scaled;

            float u = p->u;
            float v = p->v;

            float x_c = (u - cx)/fx*depth;
            float y_c = (v - cy)/fy*depth;
            float z_c = depth;

            float x_w =
                    mTwc(0,0)*x_c
                    + mTwc(0,1)*y_c
                    + mTwc(0,2)*z_c
                    + mTwc(0,3);
            float y_w =
                    mTwc(1,0)*x_c
                    + mTwc(1,1)*y_c
                    + mTwc(1,2)*z_c
                    +mTwc(1,3);
            float z_w =
                    mTwc(2,0)*x_c
                    + mTwc(2,1)*y_c
                    + mTwc(2,2)*z_c
                    + mTwc(2,3);

            Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 1> w_tmp(x_w,y_w,z_w);
            point_cloud_.push_back(w_tmp);
        }
    }
}

void dso::IOWrap::RegistrationOutputWrapper::vectorToFile() {
    std::cout << "vec to file" << std::endl;

    std::ofstream tmp_file;
    tmp_file.open(csv_point_cloud_);

    if (tmp_file.is_open()) {

        std::string x = "";
        std::string y = "";
        std::string z = "";

        for(int i=0; i < this->point_cloud_.size(); i++) {
            x.append(std::to_string(point_cloud_[i][0]) + ",");
            y.append(std::to_string(point_cloud_[i][1]) + ",");
            z.append(std::to_string(point_cloud_[i][2]) + ",");
        }

        tmp_file << x << std::endl;
        tmp_file << y << std::endl;
        tmp_file << z << std::endl;
    }

    tmp_file.close();
}

void dso::IOWrap::RegistrationOutputWrapper::labelsToFile() {
    std::cout << "labels to file" << std::endl;

    std::ofstream tmp_file;
    tmp_file.open(this->csv_seq_labels_);

    if (tmp_file.is_open()) {
        for(int i=0; i<name_label_.size(); i++) {
            tmp_file << name_label_[i].name
                     << ","
                     << name_label_[i].label
                     << ","
                     << name_label_[i].seq
                     << std::endl;
        }
    }

    tmp_file.close();
}

void dso::IOWrap::RegistrationOutputWrapper::showImgs() {
    std::cout << "\n" << "Show Imgs " << start_idx_ << std::endl;
    std::cout << "Ms " << seq_Ms_.size() << std::endl;
    std::cout << "Imgs " << seq_imgs_.size() << std::endl;

    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M_start =
            seq_Ms_.find(start_idx_)->second;
    std::vector<cv::Point> img_pts;
    this->computeImgPts(M_start, this->roi_pts_, &img_pts);

    cv::Size size_start;
    cv::Mat tmp_img;
    cv::Mat tmp_img_cropped;
//    cv::Mat tmp_img_rectified;

    for (int i=start_idx_; i < start_idx_+seq_length_; i++) {
        std::cout << i << std::endl;

        if (i==start_idx_) {
            tmp_img = seq_imgs_.find(start_idx_)->second;
            size_start = tmp_img.size();
        } else {

            std::vector<Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3>> Hs;
            this->computeH(seq_Ms_.find(i)->second, M_start, K_, &Hs);

            cv::Mat Hp;
            cv::eigen2cv(Hs[1], Hp);

            cv::warpPerspective(seq_imgs_.find(i)->second, tmp_img, Hp, size_start);
        }

//        cv::Mat Hrect = cv::findHomography(img_pts, roi_pts_rectified_, CV_RANSAC);
//        cv::warpPerspective(tmp_img, tmp_img_rectified, Hrect, size_start);

        //TODO scaling of width and height
        try {
            assert(img_pts.size()==4);
            tmp_img_cropped = tmp_img(cv::Rect(img_pts[0].x,
                                               img_pts[0].y,
                                               40,
                                               40));
        } catch (cv::Exception& e) {
            std::cout << "cropping failed" << std::endl;
        }

        if (store_imgs_) {
            this->storeImgs(tmp_img_cropped, i);
        }

        if (!nogui_) {
//        cv::imshow("Image Window Test [homography] [rectified]", tmp_img_rectified); //TODO param and add to pangolin
            cv::imshow("Image Window Test [homography] [cropped]", tmp_img_cropped); //TODO param and add to pangolin
            this->drawFilledCircle(tmp_img, img_pts);
            cv::imshow("Image Window Test [homography]", tmp_img); //TODO param and add to pangolin
            cv::waitKey(500);
        }
    }
}

void dso::IOWrap::RegistrationOutputWrapper::storeImgs(cv::Mat img, int id) {
    //TODO png to param
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    try {
        std::string filename = img_folder_+std::to_string(id)+".png";
        labels tmp_labels;
        tmp_labels.name = filename;
        tmp_labels.label = label_;
        tmp_labels.seq = seq_idx_;
        cv::imwrite(filename, img, compression_params);
        this->name_label_.push_back(tmp_labels);
    }
    catch (std::runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }
}

void dso::IOWrap::RegistrationOutputWrapper::drawFilledCircle(
        cv::Mat img, std::vector<cv::Point> centers)
{
    int thickness = -1;
    int lineType = 8;

    for (int i=0; i<centers.size(); i++) {
        cv::circle(img,
                   centers[i],
                   5,
                   cv::Scalar(0, 0, 255),
                   thickness,
                   lineType);
    }
}

//TODO convert to unit tests
void dso::IOWrap::RegistrationOutputWrapper::checkFunctionOutput() {

    //TODO unit test
    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M_15;
    M_15 << 0.998316, 0.0377052, 0.0440835, 0.000936562,
            -0.0353534, 0.99797, -0.0529626, -0.0252243,
            -0.045991, 0.0513149, 0.997623, -0.00978344,
            0, 0, 0, 1;
    std::vector<Eigen::Vector3d> wpts_15;
    this->computeWPtsInCamFrame(this->world_pts_, &wpts_15, M_15.inverse());
    std::cout << wpts_15[0] << "\n" << std::endl;
    std::cout << wpts_15[1] << "\n" << std::endl;
    std::cout << wpts_15[2] << "\n" << std::endl;
    std::cout << "\n" << std::endl;

    //TODO unit test
    std::cout << Eigen::Vector3d(0.4769, -0.7551, -0.4498) << std::endl;
    std::cout << this->computeNormalToPlane(this->world_pts_) << std::endl;
    std::cout << "\n" << std::endl;

    //TODO unit test
    std::cout << Eigen::Vector3d(-0.5235, 0.7587, 0.3877) << std::endl;
    std::cout << this->computeNormalToPlane(wpts_15) << std::endl;
    std::cout << "\n" << std::endl;

    //TODO unit test
    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M_25;
    M_25 << 0.994477, 0.0995525, 0.0332257, -0.00885576,
            -0.097898, 0.99403, -0.0481775, -0.0295688,
            -0.0378235, 0.0446587, 0.998286, -0.00738896,
            0, 0, 0, 1;
    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3> K;
    K << 277.33, 0, 312.2340,
            0, 291.4030, 239.7760,
            0, 0, 1.0000;
    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3> He;
    He << 0.9914, -0.0526, 0.0162,
            0.0585, 1.0033, -0.0010,
            -0.0094, 0.0021, 0.9987;
    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3> Hp;
    Hp << 0.9808, -0.0478, 21.5629,
            0.0534, 1.0051, -18.4634,
            -0.0000, 0.0000, 1.0077;
    std::vector<Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3>> Hs;
    this->computeH(M_15.inverse(), M_25.inverse(), K, &Hs);
    assert(((Hs[0]-He).array() <= 0.001).count() == 9);
    assert(((Hs[1]-Hp).array() <= 0.001).count() == 9);
}

Eigen::Vector3d dso::IOWrap::RegistrationOutputWrapper::computeNormalToPlane(
        const std::vector<Eigen::Vector3d> points) {
    assert(points.size() == 3);
    Eigen::Vector3d tmp_v1 = points[1] - points[0];
    Eigen::Vector3d tmp_v2 = points[0] - points[2];
    Eigen::Vector3d v = tmp_v1.cross(tmp_v2);
    return v.normalized();
}

void dso::IOWrap::RegistrationOutputWrapper::computeImgPts(
        const Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M,
        const std::vector<Eigen::Vector3d> input,
        std::vector<cv::Point>* output) {

    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 4> K_tmp =
            Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 4>::Zero(3, 4);
    K_tmp.block<3,3>(0,0) = K_;

    for (int i=0; i<input.size(); i++) {
        Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 1> tmp_wpt;
        Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 1> tmp_imgpt;
        tmp_wpt << input[i],
                1;
        tmp_imgpt = K_tmp * M * tmp_wpt;
        cv::Point tmp_p(tmp_imgpt[0]/tmp_imgpt[2], tmp_imgpt[1]/tmp_imgpt[2]);
        output->push_back(tmp_p);
    }

    assert(output->size() == input.size());
}

void dso::IOWrap::RegistrationOutputWrapper::computeWPtsInCamFrame(
        const std::vector<Eigen::Vector3d> input,
        std::vector<Eigen::Vector3d>* output,
        const Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M) {

    for (int i=0; i<input.size(); i++) {
        Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 1> tmp_wpt;
        tmp_wpt << input[i],
                1;
        tmp_wpt = M * tmp_wpt;
        output->push_back(Eigen::Vector3d(tmp_wpt[0], tmp_wpt[1], tmp_wpt[2]));
    }

    assert(output->size() == input.size());
}

void dso::IOWrap::RegistrationOutputWrapper::computeH(
        const Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M_c1,
        const Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M_c2,
        const Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3> K,
        std::vector<Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3>>* Hs) {

//    std::cout << "\n"
//              << M_c1
//              << "\n"
//              << M_c2
//              << "\n"
//              << K
//              << "\n" << std::endl;

    std::vector<Eigen::Vector3d> wpts_c1;
    this->computeWPtsInCamFrame(this->world_pts_, &wpts_c1, M_c1);
    Eigen::Vector3d vn_c1 = this->computeNormalToPlane(wpts_c1);
    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M_c1_c2 = M_c2 * M_c1.inverse();
    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3> He, Hp;

    He = M_c1_c2.block<3,3>(0,0) + (M_c1_c2.block<3,1>(0,3) * vn_c1.transpose() / wpts_c1[2].dot(vn_c1));
    Hp = K * He * K.inverse();
    Hs->push_back(He);
    Hs->push_back(Hp);

//    std::cout << "\n"
//              << Hs->at(0)
//              << "\n"
//              << Hs->at(1)
//              << "\n"
//              << std::endl;
}
