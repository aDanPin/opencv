// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "../test_precomp.hpp"

#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

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

    TEST(GenericGraph, NoRecompileWithSameMeta3)
    {
        cv::GComputation cc([&](){
            cv::Size szOut(4, 4);

            cv::GMat in1;
            cv::GProtoInputArgs ins = GIn(in1);

            cv::GMat in2;
            ins += GIn(in2);

            cv::GMat out1 = cv::gapi::resize(in1, szOut);
            cv::GProtoOutputArgs outs = GOut(out1);

            cv::GMat out2 = cv::gapi::resize(in2, szOut);
            outs += GOut(out2);

            return cv::GComputation(std::move(ins), std::move(outs));
        });

        EXPECT_NO_THROW(cc.compileStreaming(cv::compile_args(cv::gapi::core::cpu::kernels())));
    }

    TEST(GenericGraph, NoRecompileWithSameMeta4)
    {
        cv::Size szOut(4, 4);
        cv::GComputation cc([&](){
            cv::GMat in1;
            cv::GProtoInputArgs ins = GIn(in1);

            cv::GMat in2;
            ins += GIn(in2);

            cv::GMat out1 = cv::gapi::resize(in1, szOut);
            cv::GProtoOutputArgs outs = GOut(out1);

            cv::GMat out2 = cv::gapi::resize(in2, szOut);
            outs += GOut(out2);

            return cv::GComputation(std::move(ins), std::move(outs));
        });

        // G-API test code
        cv::Mat in_mat1( 8,  8, CV_8UC3);
        cv::Mat in_mat2(16, 16, CV_8UC3);
        cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
        cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));

        cv::Mat out_mat1, out_mat2;
        cv::GRunArgsP out_vector;
        out_vector += cv::gout(out_mat1);
        out_vector += cv::gout(out_mat2);

        auto stream = cc.compileStreaming(cv::compile_args(cv::gapi::core::cpu::kernels()));
        stream.setSource(gin(in_mat1, in_mat2));

        stream.start();
        stream.pull(std::move(out_vector));
        stream.stop();

        // OCV ref code
        cv::Mat cv_out_mat1, cv_out_mat2;
        cv::resize(in_mat1, cv_out_mat1, szOut);
        cv::resize(in_mat2, cv_out_mat2, szOut);

        EXPECT_EQ(0, cvtest::norm(out_mat1, cv_out_mat1, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat2, cv_out_mat2, NORM_INF));
    }
} // namespace opencv_test