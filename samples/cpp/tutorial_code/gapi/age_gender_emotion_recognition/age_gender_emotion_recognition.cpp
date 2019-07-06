#include "opencv2/opencv_modules.hpp"
#if defined(HAVE_OPENCV_GAPI) && defined(HAVE_INF_ENGINE)

#include <chrono>
#include <iomanip>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"
#include "opencv2/gapi/infer.hpp"
#include "opencv2/gapi/infer/ie.hpp"
#include "opencv2/gapi/cpu/gcpukernel.hpp"

namespace {
const std::string about =
    "This is an OpenCV-based version of Security Barrier Camera example";
const std::string keys =
    "{ h help |   | print this help message }"
    "{ input  |   | Path to an input video file }"
    "{ fdm    |   | IE face detection model IR }"
    "{ fdw    |   | IE face detection model weights }"
    "{ fdd    |   | IE face detection device }"
    "{ agem   |   | IE age/gender recognition model IR }"
    "{ agew   |   | IE age/gender recognition model weights }"
    "{ aged   |   | IE age/gender recognition model device }"
    "{ emom   |   | IE emotions recognition model IR }"
    "{ emow   |   | IE emotions recognition model weights }"
    "{ emod   |   | IE emotions recognition model device }"
    "{ pure   |   | When set, no output is displayed. Useful for benchmarking }";

struct Avg {
    struct Elapsed {
        explicit Elapsed(double ms) : ss(ms/1000.), mm(ss/60) {}
        const double ss;
        const int    mm;
    };

    using MS = std::chrono::duration<double, std::ratio<1, 1000>>;
    using TS = std::chrono::time_point<std::chrono::high_resolution_clock>;
    TS started;

    void    start() { started = now(); }
    TS      now() const { return std::chrono::high_resolution_clock::now(); }
    double  tick() const { return std::chrono::duration_cast<MS>(now() - started).count(); }
    Elapsed elapsed() const { return Elapsed{tick()}; }
    double  fps(std::size_t n) const { return static_cast<double>(n) / (tick() / 1000.); }
};
std::ostream& operator<<(std::ostream &os, const Avg::Elapsed &e) {
    os << e.mm << ':' << (e.ss - 60*e.mm);
    return os;
}
} // namespace

namespace custom {
GAPI_NETWORK(Faces,      <cv::GMat(cv::GMat)>, "face-detector");
                                                 
using AGInfo = std::tuple<cv::GMat, cv::GMat>;   
GAPI_NETWORK(AgeGender,  <AGInfo(cv::GMat)>,   "age-gender-recoginition");
GAPI_NETWORK(Emotions,   <cv::GMat(cv::GMat)>, "emotions-recognition");

GAPI_OPERATION(PostProc, <cv::GArray<cv::Rect>(cv::GMat, cv::GMat)>, "custom.fd_postproc") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVPostProc, PostProc) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Mat &in_frame,
                    std::vector<cv::Rect> &out_faces) {
        const int MAX_PROPOSALS = 200;
        const int OBJECT_SIZE   =   7;
        const cv::Size upscale = in_frame.size();
        const cv::Rect surface({0,0}, upscale);

        out_faces.clear();

        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++) {
            const float image_id   = data[i * OBJECT_SIZE + 0]; // batch id
            const float label      = data[i * OBJECT_SIZE + 1];
            const float confidence = data[i * OBJECT_SIZE + 2];
            const float rc_left    = data[i * OBJECT_SIZE + 3];
            const float rc_top     = data[i * OBJECT_SIZE + 4];
            const float rc_right   = data[i * OBJECT_SIZE + 5];
            const float rc_bottom  = data[i * OBJECT_SIZE + 6];

            if (image_id < 0.f) {  // indicates end of detections
                break;
            }
            if (confidence < 0.5f) { // fixme: hard-coded snapshot
                continue;
            }

            cv::Rect rc;
            rc.x      = static_cast<int>(rc_left   * upscale.width);
            rc.y      = static_cast<int>(rc_top    * upscale.height);
            rc.width  = static_cast<int>(rc_right  * upscale.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * upscale.height) - rc.y;
            out_faces.push_back(rc);
        }
    }
};
} // namespace custom

namespace labels {
const std::string genders[] = {
    "Female", "Male"
};
const std::string emotions[] = {
    "neutral", "happy", "sad", "surprise", "anger"
};

void DrawResults(cv::Mat &frame,
                 const std::vector<cv::Rect> &faces,
                 const std::vector<cv::Mat>  &out_ages,
                 const std::vector<cv::Mat>  &out_genders,
                 const std::vector<cv::Mat>  &out_emotions) {
    CV_Assert(faces.size() == out_ages.size());
    CV_Assert(faces.size() == out_genders.size());
    CV_Assert(faces.size() == out_emotions.size());

    for (auto it = faces.begin(); it != faces.end(); ++it) {
        const auto idx = std::distance(faces.begin(), it);
        const auto &rc = *it;

        const float *ages_data     = out_ages[idx].ptr<float>();
        const float *genders_data  = out_genders[idx].ptr<float>();
        const float *emotions_data = out_emotions[idx].ptr<float>();
        const auto gen_id = std::max_element(genders_data,  genders_data  + 2) - genders_data;
        const auto emo_id = std::max_element(emotions_data, emotions_data + 5) - emotions_data;

        std::stringstream ss;
        ss << static_cast<int>(ages_data[0]*100)
           << ' '
           << genders[gen_id]
           << ' '
           << emotions[emo_id];

        const int ATTRIB_OFFSET = 15;
        cv::rectangle(frame, rc, {0, 255, 0},  4);
        cv::putText(frame, ss.str(),
                    cv::Point(rc.x, rc.y - ATTRIB_OFFSET),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    cv::Scalar(0, 0, 255));
    }
}

void DrawFPS(cv::Mat &frame, std::size_t n, double fps) {
    std::ostringstream out;
    out << "FRAME " << n << ": "
        << std::fixed << std::setprecision(2) << fps
        << " FPS (AVG)";
    cv::putText(frame, out.str(),
                cv::Point(0, frame.rows),
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                cv::Scalar(0, 0, 0),
                2);
}
} // namespace labels

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const std::string input = cmd.get<std::string>("input");
    const bool no_show = cmd.get<bool>("pure");

    cv::GComputation pp([]() {
            cv::GMat in;
            cv::GMat detections           = cv::gapi::infer<custom::Faces>(in);
            cv::GArray<cv::Rect> faces    = custom::PostProc::on(detections, in);
            cv::GArray<cv::GMat> ages;
            cv::GArray<cv::GMat> genders;
            std::tie(ages, genders)       = cv::gapi::infer<custom::AgeGender>(faces, in);
            cv::GArray<cv::GMat> emotions = cv::gapi::infer<custom::Emotions>(faces, in);
            cv::GMat frame = cv::gapi::copy(in); // pass-through the input frame
            return cv::GComputation(cv::GIn(in),
                                    cv::GOut(frame, faces, ages, genders, emotions));
        });

    // Note: it might be very useful to have dimensions loaded at this point!
    auto det_net = cv::gapi::ie::Params<custom::Faces> {
        cmd.get<std::string>("fdm"),   // path to topology IR
        cmd.get<std::string>("fdw"),   // path to weights
        cmd.get<std::string>("fdd"),   // device specifier
    };

    auto age_net = cv::gapi::ie::Params<custom::AgeGender> {
        cmd.get<std::string>("agem"),   // path to topology IR
        cmd.get<std::string>("agew"),   // path to weights
        cmd.get<std::string>("aged"),   // device specifier
    }.cfgOutputLayers({ "age_conv3", "prob" });

    auto emo_net = cv::gapi::ie::Params<custom::Emotions> {
        cmd.get<std::string>("emom"),   // path to topology IR
        cmd.get<std::string>("emow"),   // path to weights
        cmd.get<std::string>("emod"),   // device specifier
    };

    auto kernels = cv::gapi::kernels<custom::OCVPostProc>();
    auto networks = cv::gapi::networks(det_net, age_net, emo_net);
    auto cc = pp.compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size(1920,1080)},
                                  cv::compile_args(kernels, networks));

    std::cout << "Reading " << input << std::endl;
    cc.setSource(cv::gapi::GVideoCapture{input});

    Avg avg;
    avg.start();
    cc.start();

    cv::Mat frame;
    std::vector<cv::Rect> faces;
    std::vector<cv::Mat> out_ages;
    std::vector<cv::Mat> out_genders;
    std::vector<cv::Mat> out_emotions;
    std::size_t frames = 0u;

    // Implement different execution policies depending on the display option
    // for the best performance.
    while (cc.running()) {
        auto out_vector = cv::gout(frame, faces, out_ages, out_genders, out_emotions);
        if (no_show) {
            // This is purely a video processing. No need to balance with UI rendering.
            // Use a blocking pull() to obtain data. Break the loop if the stream is over.
            if (!cc.pull(std::move(out_vector)))
                break;
        } else if (!cc.try_pull(std::move(out_vector))) {
            // Use a non-blocking try_pull() to obtain data.
            // If there's no data, let UI refresh (and handle keypress)
            if (cv::waitKey(1) >= 0) break;
            else continue;
        }
        // At this point we have data for sure (obtained in either blocking or non-blocking way).
        frames++;
        labels::DrawResults(frame, faces, out_ages, out_genders, out_emotions);
        labels::DrawFPS(frame, frames, avg.fps(frames));
        if (!no_show) cv::imshow("Out", frame);
    }
    cc.stop();
    std::cout << "Processed " << frames << " frames in " << avg.elapsed() << std::endl;

    return 0;
}
#else
#include <iostream>
int main()
{
    std::cerr << "This tutorial code requires G-API module "
                 "with Inference Engine backend to run"
              << std::endl;
    return 1;
}
#endif  // HAVE_OPECV_GAPI && HAVE_INF_ENGINE
