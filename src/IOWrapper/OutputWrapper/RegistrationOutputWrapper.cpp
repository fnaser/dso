//
// Created by fnaser on 10/10/18.
//

#include "IOWrapper/OutputWrapper/RegistrationOutputWrapper.h"

dso::IOWrap::RegistrationOutputWrapper::RegistrationOutputWrapper()
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

    //TODO unit test
    std::cout << Eigen::Vector3d(0.4769, -0.7551, -0.4498) << std::endl;
    std::cout << this->computeNormalToPlane(this->world_pts_) << std::endl;
    std::cout << "\n" << std::endl;

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
    this->computeH(M_15.inverse(), M_25.inverse(), K);
}

void dso::IOWrap::RegistrationOutputWrapper::publishCamPose(dso::FrameShell* frame,
                                                            dso::CalibHessian* HCalib) {

}

void dso::IOWrap::RegistrationOutputWrapper::pushLiveFrame(FrameHessian* image) {

}

Eigen::Vector3d dso::IOWrap::RegistrationOutputWrapper::computeNormalToPlane(
        const std::vector<Eigen::Vector3d> points) {
    assert(points.size() == 3);
    Eigen::Vector3d tmp_v1 = points[1] - points[0];
    Eigen::Vector3d tmp_v2 = points[0] - points[2];
    Eigen::Vector3d v = tmp_v1.cross(tmp_v2); //TODO check direction
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
        const Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3> K) {

    std::vector<Eigen::Vector3d> wpts_c1;
    this->computeWPtsInCamFrame(this->world_pts_, &wpts_c1, M_c1);
    Eigen::Vector3d vn_c1 = this->computeNormalToPlane(wpts_c1);
    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M_c1_c2 = M_c2 * M_c1.inverse();
    Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3> He, Hp;

    He = M_c1_c2.block<3,3>(0,0) + (M_c1_c2.block<3,1>(0,3) * vn_c1.transpose() / wpts_c1[2].dot(vn_c1));
    std::cout << He << std::endl; //TODO

    Hp = K * He * K.inverse();
    std::cout << Hp << std::endl; //TODO
}
