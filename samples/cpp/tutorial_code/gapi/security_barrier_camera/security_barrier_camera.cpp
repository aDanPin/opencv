#include "opencv2/opencv_modules.hpp"
#if defined(HAVE_OPENCV_GAPI) && defined(HAVE_INF_ENGINE)

#include <chrono>
#include <iomanip>

#include <inference_engine.hpp>
#include <ie_iextension.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"
#include "opencv2/gapi/util/optional.hpp"
#include "opencv2/highgui.hpp"

const std::string about =
    "This is an OpenCV-based version of Security Barrier Camera example";
const std::string keys =
    "{ h help |   | print this help message }"
    "{ input  |   | Path to an input video file }"
    "{ detm   |   | IE vehicle/license plate detection model IR }"
    "{ detw   |   | IE vehicle/license plate detection model weights }"
    "{ vehm   |   | IE vehicle attributes model IR }"
    "{ vehw   |   | IE vehicle attributes model weights }"
    "{ lprm   |   | IE license plate recognition IR }"
    "{ lprw   |   | IE license plate recognition model weights }";

namespace IE = InferenceEngine;

namespace {
// Taken from IE samples
IE::Blob::Ptr wrapMatToBlob(const cv::Mat &mat) {
    size_t channels = mat.channels();
    size_t height = mat.size().height;
    size_t width = mat.size().width;

    size_t strideH = mat.step.buf[0];
    size_t strideW = mat.step.buf[1];

    bool is_dense =
            strideW == channels &&
            strideH == channels * width;

    if (!is_dense) THROW_IE_EXCEPTION
                << "Doesn't support conversion from not dense cv::Mat";

    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                      {1, channels, height, width},
                                      InferenceEngine::Layout::NHWC);

    return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
}

inline IE::ROI toROI(const cv::Rect &rc) {
    return IE::ROI
        { 0u
        , static_cast<std::size_t>(rc.x)
        , static_cast<std::size_t>(rc.y)
        , static_cast<std::size_t>(rc.width)
        , static_cast<std::size_t>(rc.height)
        };
}

struct Timer {
    using MS = std::chrono::duration<double, std::ratio<1, 1000>>;

    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::size_t n = 0u;
    float thisAvg = 0.f;

    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    MS stop() {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto result  = std::chrono::duration_cast<MS>(endTime - startTime);

        float past_n = static_cast<float>(n++);
        float dur    = static_cast<float>(result.count());
        thisAvg      = (past_n / n) * thisAvg + dur / n;

        return result;
    }
};
} // namespace

struct NetDesc {
    std::string topo;
    std::string weights;
};

struct Net {
    IE::CNNNetwork net;

    std::string def_input;
    std::string def_output;

    explicit Net(const NetDesc &d) {
        IE::CNNNetReader reader;
        reader.ReadNetwork(d.topo);
        reader.ReadWeights(d.weights);
        net = reader.getNetwork();
        def_input = net.getInputsInfo().begin()->first;
        def_output = net.getOutputsInfo().begin()->first;
    };

    void setupIn(const IE::Precision &prec,
                 const IE::Layout &layout,
                 const IE::ResizeAlgorithm &res,
                 const std::string &layer = {}) {
        const auto this_layer = layer.empty() ? def_input : layer;
        auto& ii = net.getInputsInfo().at(this_layer);
        ii->getPreProcess().setResizeAlgorithm(res);
        ii->setLayout(layout);
        ii->setPrecision(prec);
    }

    void setupIn(const IE::Precision &prec,
                 const std::string &layer = {}) {
        const auto this_layer = layer.empty() ? def_output : layer;
        auto &oi = net.getOutputsInfo().at(this_layer);
        oi->setPrecision(prec);
    }
};

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const std::string input = cmd.get<std::string>("input");
    const NetDesc
        det_desc { cmd.get<std::string>("detm"), cmd.get<std::string>("detw") },
        veh_desc { cmd.get<std::string>("vehm"), cmd.get<std::string>("vehw") },
        lpr_desc { cmd.get<std::string>("lprm"), cmd.get<std::string>("lprw") };
    if (!cmd.check()) {
        cmd.printErrors();
        return 1;
    }

    std::cout << "Reading " << input << std::endl;
    cv::VideoCapture cap(input);

    Net det(det_desc), veh(veh_desc), lpr(lpr_desc);
    det.setupIn(IE::Precision::U8, IE::Layout::NCHW, IE::RESIZE_BILINEAR);
    veh.setupIn(IE::Precision::U8, IE::Layout::NCHW, IE::RESIZE_BILINEAR);
    lpr.setupIn(IE::Precision::U8, IE::Layout::NCHW, IE::RESIZE_BILINEAR);

    // FIXME: Achtung: R1 API
    IE::InferencePlugin plugin = IE::PluginDispatcher().getPluginByDevice("GPU");

    IE::ExecutableNetwork det_exec = plugin.LoadNetwork(det.net, {});
    IE::ExecutableNetwork veh_exec = plugin.LoadNetwork(veh.net, {});
    IE::ExecutableNetwork lpr_exec = plugin.LoadNetwork(lpr.net, {});

    IE::InferRequest det_rq = det_exec.CreateInferRequest();
    IE::InferRequest veh_rq = veh_exec.CreateInferRequest();
    IE::InferRequest lpr_rq = lpr_exec.CreateInferRequest();

    IE::Blob::Ptr seq_ind = lpr_rq.GetBlob("seq_ind");
    float *seq_ind_data = seq_ind->buffer().as<float*>();
    seq_ind_data[0] = 0.f;
    for (int i = 1; i < 88; i++) {
        seq_ind_data[i] = 1.f;
    }

    const int MAX_PROPOSALS = 200;
    const int MAX_LICENSE   =  88;
    const int OBJECT_SIZE   =   7;
    const std::string colors[] = {
        "white", "gray", "yellow", "red", "green", "blue", "black"
    };
    const std::string types[] = {
        "car", "van", "truck", "bus"
    };
    const std::vector<std::string> license_text = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
        "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
        "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
        "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
        "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
        "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
        "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
        "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
        "<Zhejiang>", "<police>",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z"
    };

    cv::Mat frame;
    std::vector<cv::Rect> vehicles, plates;
    std::vector<std::pair<std::string, std::string> > veh_attrs;
    std::vector<std::string> licenses;

    Timer t;
    while (cv::waitKey(1) < 0) {
        t.start();
        cap >> frame;
        if (frame.empty())
            break;
        const cv::Size upscale = frame.size();
        const cv::Rect surface({0,0}, upscale);

        IE::Blob::Ptr input_frame = wrapMatToBlob(frame);
        det_rq.SetBlob(det.def_input, input_frame);
        det_rq.Infer();

        vehicles.clear();
        plates.clear();
        veh_attrs.clear();
        licenses.clear();
        const float *data = det_rq.GetBlob(det.def_output)->buffer().as<float*>();
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

            using PT = cv::Point;
            using SZ = cv::Size;
            switch (static_cast<int>(label)) {
            case 1: vehicles.push_back(rc & surface); break;
            case 2: plates.emplace_back((rc-PT(5,5)+SZ(10,10)) & surface); break;
            default: THROW_IE_EXCEPTION << "Impossible happened";
            }
        }

        for (auto &&rc: vehicles) {
            IE::Blob::Ptr veh_roi = IE::make_shared_blob(input_frame, toROI(rc));
            veh_rq.SetBlob(veh.def_input, veh_roi);
            veh_rq.Infer();

            const float *clr_data = veh_rq.GetBlob("color")->buffer().as<float*>();
            const float *cls_data = veh_rq.GetBlob("type" )->buffer().as<float*>();
            const auto color_id = std::max_element(clr_data, clr_data + 7) - clr_data;;
            const auto  type_id = std::max_element(cls_data, cls_data + 4) - cls_data;
            veh_attrs.emplace_back(colors[color_id], types[type_id]);
        }

        for (auto &&rc: plates) {
            IE::Blob::Ptr plt_roi = IE::make_shared_blob(input_frame, toROI(rc));
            lpr_rq.SetBlob(lpr.def_input, plt_roi);
            lpr_rq.Infer();

            std::string result;
            const auto *lpr_data = lpr_rq.GetBlob(lpr.def_output)->buffer().as<float*>();
            for (int i = 0; i < MAX_LICENSE; i++) {
                if (lpr_data[i] == -1)
                    break;
                result += license_text[static_cast<size_t>(lpr_data[i])];
            }
            licenses.push_back(std::move(result));
        }

        for (auto it = vehicles.begin(); it != vehicles.end(); ++it) {
            const auto &rc   = *it;
            const auto &attr = veh_attrs[ std::distance(vehicles.begin(), it) ];

            const int ATTRIB_OFFSET = 25;
            cv::rectangle(frame, rc, {0, 255, 0},  4);
            cv::putText(frame, attr.first,
                        cv::Point(rc.x + 5, rc.y + ATTRIB_OFFSET),
                        cv::FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        cv::Scalar(255, 0, 0));
            cv::putText(frame, attr.second,
                        cv::Point(rc.x + 5, rc.y + ATTRIB_OFFSET * 2),
                        cv::FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        cv::Scalar(255, 0, 0));
        }

        for (auto it = plates.begin(); it != plates.end(); ++it) {
            const auto &rc   = *it;
            const auto &text = licenses[ std::distance(plates.begin(), it) ];
            const int LPR_OFFSET = 50;
            const int y_pos = std::max(0, rc.y + rc.height - LPR_OFFSET);

            cv::rectangle(frame, rc, {0, 0, 255},  4);
            cv::putText(frame, text,
                        cv::Point(rc.x, y_pos),
                        cv::FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        cv::Scalar(0, 0, 255));
        }
        t.stop();

        std::ostringstream out;
        out << "FRAME " << t.n << ": "
            << std::fixed << std::setprecision(2) << 1000. / t.thisAvg
            << " FPS";
        cv::putText(frame, out.str(),
                    cv::Point(0, upscale.height),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1,
                    cv::Scalar(0, 0, 0),
                    2);

        cv::imshow("Output", frame);
    }
    return 0;
}
#else
#error Not this time bro
#endif  // HAVE_OPECV_GAPI && HAVE_INF_ENGINE
