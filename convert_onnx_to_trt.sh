sudo pip3 install onnx-simplifier
python3 -m onnxsim retinaface_mbv2.onnx retinaface_mbv2_sim.onnx
onnx2trt retinaface_mbv2_sim.onnx -o retinaface.trt
