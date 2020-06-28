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

        EXPECT_NO_THROW(cc.apply(in_mat1, out_mat1, out_mat1));
    }

} // namespace opencv_test