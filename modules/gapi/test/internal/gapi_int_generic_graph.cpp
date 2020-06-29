// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "../test_precomp.hpp"
//#include "api/gcomputation_priv.hpp"

#include <opencv2/gapi/fluid/gfluidkernel.hpp> // ??
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>

namespace opencv_test
{
    TEST(GenericGraph, NoRecompileWithSameMeta)
    {
        cv::GComputation cc([]() {
            cv::GMat in;
            cv::GMat out1 = cv::gapi::copy(in);
            cv::GProtoOutputArgs outs = GOut(out1);

            cv::GMat out2 = cv::gapi::copy(in);
            outs += GOut(out2);

            return cv::GComputation(cv::GIn(in), std::move(outs));
        });

        cv::Mat in_mat1 = cv::Mat::eye(32, 32, CV_8UC1);
        cv::Mat out_mat1;
        cv::Mat out_mat2;

        EXPECT_NO_THROW(cc.apply(cv::gin(in_mat1), cv::gout(out_mat1, out_mat1)));
    }

    TEST(GenericGraph, NoRecompileWithSameMeta1)
    {
        cv::GComputation cc([]() {
            cv::GMat in1;
            cv::GProtoInputArgs ins = GIn(in1);

            cv::GMat in2;
            ins += GIn(in2);

            cv::GMat out = cv::gapi::copy(in1 + in2);

            return cv::GComputation(std::move(ins), GOut(out));
        });

        cv::Mat in_mat1 = cv::Mat::eye(32, 32, CV_8UC1);
        cv::Mat in_mat2 = cv::Mat::eye(32, 32, CV_8UC1);
        cv::Mat out_mat;

        EXPECT_NO_THROW(cc.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat)));
    }

    TEST(GenericGraph, NoRecompileWithSameMeta2)
    {
        cv::GComputation cc([]() {
            cv::GMat in1;
            cv::GProtoInputArgs ins = GIn(in1);

            cv::GMat in2;
            ins += GIn(in2);

            cv::GMat out1 = cv::gapi::copy(in1 + in2);
            cv::GProtoOutputArgs outs = GOut(out1);

            cv::GMat out2 = cv::gapi::copy(in1 + in2);
            outs += GOut(out2);

            return cv::GComputation(std::move(ins), std::move(outs));
        });

        cv::Mat in_mat1 = cv::Mat::eye(32, 32, CV_8UC1);
        cv::Mat in_mat2 = cv::Mat::eye(32, 32, CV_8UC1);
        cv::Mat out_mat1, out_mat2;

        EXPECT_NO_THROW(cc.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat1, out_mat2)));
    }
} // namespace opencv_test