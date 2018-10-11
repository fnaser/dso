/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"

#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>

namespace dso
{

    class FrameHessian;
    class CalibHessian;
    class FrameShell;

    namespace IOWrap
    {

        class RegistrationOutputWrapper : public Output3DWrapper
        {

        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            RegistrationOutputWrapper(int, int);

            virtual ~RegistrationOutputWrapper()
            {
                printf("OUT: Destroyed SampleOutputWrapper\n");
            }

            virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>,
                    Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override {}

            virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override {}

            virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override;

            virtual void pushLiveFrame(FrameHessian* image) override;

            virtual void pushDepthImage(MinimalImageB3* image) override {}

            virtual bool needPushDepthImage() override {
                return false;
            }

            virtual void pushDepthImageFloat(MinimalImageF* image, FrameHessian* KF ) override {}

        private:

            int h_, w_;
            int start_idx_, seq_length_;

            std::vector<Eigen::Vector3d> world_pts_;
            std::map<int, cv::Mat> seq_imgs_;
            std::map<int, Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4>> seq_Ms_;
            boost::mutex seq_Ms_mutex_;
            Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3> K_;

            Eigen::Vector3d computeNormalToPlane(
                    std::vector<Eigen::Vector3d>  points);

            void computeWPtsInCamFrame(
                    const std::vector<Eigen::Vector3d> input,
                    std::vector<Eigen::Vector3d>* output,
                    const Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M);

            void computeH(
                    const Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M_c1,
                    const Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 4, 4> M_c2,
                    const Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3> K,
                    std::vector<Eigen::Matrix<Sophus::SE3Group<double>::Scalar, 3, 3>>* Hs);

            void checkFunctionOutput();

            void showImgs();

            void storeImgs(cv::Mat img, int id);
        };
    }
}
