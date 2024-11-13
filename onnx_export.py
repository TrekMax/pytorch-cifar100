import linger
import torch
import torch.onnx
import torch.nn as nn
from models.resnet import resnet18

def main(checkpoint_path: str, onnx_path: str, net, dummy_input):
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()
    out = net(dummy_input)
    with torch.no_grad():
        torch.onnx.export(net, dummy_input, onnx_path,
                            opset_version=11,
                            input_names=["input"],
                            output_names=["output"])

if __name__ == '__main__':
    checkpoint_path = "./checkpoint/resnet18/Tuesday_12_November_2024_10h_32m_42s/resnet18-30-regular.pth"
    onnx_path = "linger_resnet18.onnx"
    net = resnet18().cuda()
    dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True).cuda()
    # print(dummy_input)
    replace_tuple = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.AvgPool2d)
    linger.trace_layers(net, net, dummy_input, fuse_bn=True)
    net = linger.init(net, quant_modules=replace_tuple, mode=linger.QuantMode.QValue)

    main(checkpoint_path, onnx_path, net, dummy_input)
