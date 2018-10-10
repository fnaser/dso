//
// Created by fnaser on 10/10/18.
//

#include "IOWrapper/OutputWrapper/RegistrationOutputWrapper.h"

dso::IOWrap::RegistrationOutputWrapper::RegistrationOutputWrapper()
{
    printf("OUT: Created SampleOutputWrapper\n");

    //TODO params
    this->world_pts_.push_back(
            Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 1>(
                    -1.5977, -2.8942, 4.7679));
    this->world_pts_.push_back(
            Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 1>(
                    -3.2313, -2.9793, 3.1786));
    this->world_pts_.push_back(
            Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 1>(
                    0.3451, 0.4708, 1.1789));

    assert(this->world_pts_.size() == 3);
}

void dso::IOWrap::RegistrationOutputWrapper::publishCamPose(dso::FrameShell* frame,
                                                            dso::CalibHessian* HCalib) {

}

void dso::IOWrap::RegistrationOutputWrapper::pushLiveFrame(FrameHessian* image) {

}