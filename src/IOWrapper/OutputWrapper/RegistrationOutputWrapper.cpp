//
// Created by fnaser on 10/10/18.
//

#include "IOWrapper/OutputWrapper/RegistrationOutputWrapper.h"

dso::IOWrap::RegistrationOutputWrapper::RegistrationOutputWrapper(int w, int h)
{
    printf("OUT: Created SampleOutputWrapper\n");

    //TODO params for seq 28
    this->world_pts_.push_back(
            Eigen::Vector3d(-1.5977, -2.8942, 4.7679));
    this->world_pts_.push_back(
            Eigen::Vector3d(-3.2313, -2.9793, 3.1786));
    this->world_pts_.push_back(
            Eigen::Vector3d(0.3451, 0.4708, 1.1789));

    assert(this->world_pts_.size() == 3);

    this->w_ = w;
    this->h_ = h;
    this->seq_length_ = 10; //TODO param
    this->start_idx_ = -1;

    checkFunctionOutput();
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

    if (start_idx_ == -1 || seq_Ms_.empty()) {
        start_idx_ = frame->id;
    }

    if (seq_Ms_.find(start_idx_) != seq_Ms_.end() &&
        seq_Ms_.find(start_idx_+seq_length_) != seq_Ms_.end() &&
        seq_imgs_.find(start_idx_) != seq_imgs_.end() &&
        seq_imgs_.find(start_idx_+seq_length_) != seq_imgs_.end()) {
        this->showImgs();
        seq_Ms_.erase(seq_Ms_.begin(), seq_Ms_.find(start_idx_));
        seq_imgs_.erase(seq_imgs_.begin(), seq_imgs_.find(start_idx_));
        start_idx_ = -1;
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
}

void dso::IOWrap::RegistrationOutputWrapper::showImgs() {
    std::cout << "Show Imgs " << start_idx_ << std::endl;
    std::cout << "Ms " << seq_Ms_.size() << std::endl;
    std::cout << "Imgs " << seq_imgs_.size() << std::endl;

    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M_start =
            seq_Ms_.find(start_idx_)->second;
    cv::Mat img_start = seq_imgs_.find(start_idx_)->second;
    cv::Size size_start = img_start.size();
    std::string cvwname = "Image Window Test [homography]"; //TODO param

    cv::imshow(cvwname, img_start);
    cv::waitKey(1000);

    for (int i=start_idx_+1; i < start_idx_+seq_length_; i++) {
        std::cout << i << std::endl;

        std::vector<Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3>> Hs;
        this->computeH(seq_Ms_.find(i)->second, M_start, K_, &Hs);

        cv::Mat Hp;
        cv::eigen2cv(Hs[1], Hp);
//        cv::Mat He;
//        cv::eigen2cv(Hs[0], He);

//        std::cout << Hp << std::endl;
//        std::cout << Hs[1] << std::endl;

        cv::Mat tmp_img;
        cv::warpPerspective(seq_imgs_.find(i)->second, tmp_img, Hp, size_start);

        cv::imshow(cvwname, tmp_img);
        cv::waitKey(500);

        this->storeImgs(tmp_img, i);
    }
}

void dso::IOWrap::RegistrationOutputWrapper::storeImgs(cv::Mat img, int id) {
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    try {
        cv::imwrite("/home/fnaser/Pictures/testing_dso_H/"+std::to_string(id)+".png", img, compression_params); //TODO param
    }
    catch (std::runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
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
