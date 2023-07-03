import torch
from collections import OrderedDict
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channel = 3
        self.out_channel = 32
        self.backbone = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.in_channel, self.out_channel, 3, 1, 1)),
            ('bn', nn.BatchNorm2d(self.out_channel)),
            ('relu', nn.ReLU(inplace=True)),
            ('maxpool', nn.MaxPool2d(3, stride=2, padding=1))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(32768, 128)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(128, 3)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = torch.flatten(x, 1)
        output = self.classifier(x)
        return output


def model_converter():
    model = Model()
    model.load_state_dict(
        torch.load("../work_dirs_switch/model_epoch_2.pth", map_location=torch.device('cpu'))['model_state_dict'])

    model.to(device)  # 这里保存的是完整模型
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 64, device=device)
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model, dummy_input, 'model_switch.onnx',
                      export_params=True,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)


model_converter()
