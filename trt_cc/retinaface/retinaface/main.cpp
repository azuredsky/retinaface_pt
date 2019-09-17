#include <RetinaFace.h>
#include <iostream>
#include "thor/os.h"
#include "thor/timer.h"
#include "timer.h"

using namespace std;

int main(int argc, char** argv) {
  string path = "../model";
  RetinaFace* rf = new RetinaFace(path, "net3");

  if (thor::os::isdir(argv[1])) {
    vector<cv::String> fn;
    string image_file_ptn = thor::os::join(argv[1], "*.jpg");
    cv::glob(image_file_ptn, fn, true);
    cout << "found all " << fn.size() << " images.\n";
    vector<Mat> imgs;
    for (auto f : fn) {
      cv::Mat src = cv::imread(f);
      imgs.push_back(src.clone());
    }

    float time = 0;
    int count = 0;

    thor::Timer timer(20);
    timer.on();

    int i = 0;
    for (auto img : imgs) {
      i++;
      cout << "inference on img: " << i << endl;
      timer.lap();
      rf->detect(img, 0.5);
      double cost = timer.lap();
      cout << "cost time: " << cost << "s"
           << " fps: " << 1 / cost << endl;
    }
  } else {
    // inference on video
    string f_suffix = thor::os::suffix(argv[1]);
    if (f_suffix == "mp4") {
      // solve video
      cv::VideoCapture cap(argv[1]);
      if (!cap.isOpened()) {
        cerr << "video open failed: " << argv[1] << endl;
        return -1;
      }

      thor::Timer timer(20);
      timer.on();

      for (;;) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        timer.lap();
        rf->detect(frame, 0.5);
        double cost = timer.lap();
        cout << "cost time: " << cost << "s"
             << " fps: " << 1 / cost << endl;

        cv::imshow("rr", frame);
        cv::waitKey(1);
      }
    }

    return 0;
  }
}
