// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_PERF_TESTS_HPP
#define OPENCV_GAPI_RENDER_PERF_TESTS_HPP



#include "../../test/common/gapi_tests_common.hpp"
#include <opencv2/gapi/imgproc.hpp>

#include <string>

namespace opencv_test
{

  using namespace perf;

  //------------------------------------------------------------------------------

    class RenderTestTexts : public TestPerfParams<tuple<std::string, cv::Size>> {};
    class RenderTestRects : public TestPerfParams<tuple<cv::Size>> {};
    class RenderTestCircles : public TestPerfParams<tuple<cv::Size>> {};
    class RenderTestLines : public TestPerfParams<tuple<cv::Size>> {};
    class RenderTestMosaics : public TestPerfParams<tuple<cv::Size>> {};
    class RenderTestImages : public TestPerfParams<tuple<cv::Size>> {};
    class RenderTestPolylines : public TestPerfParams<tuple<cv::Size>> {};

} // opencv_test

#endif // OPENCV_GAPI_RENDER_PERF_TESTS_HPP
