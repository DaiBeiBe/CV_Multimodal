import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        feature1 = x

        x = self.conv2(x)
        x = torch.relu(x)
        feature2 = x

        x = self.conv3(x)
        x = torch.relu(x)
        feature3 = x

        return feature1, feature2, feature3


if __name__ == "__main__":
    model = SimpleCNN()
    zero_input = torch.zeros(1, 1, 32, 32)
    point_input = zero_input.clone()
    point_input[0, 0, 16, 16] = 1

    with torch.no_grad():
        zero_features = model(zero_input)
        point_features = model(point_input)

    for index, (zero_feature, point_feature) in enumerate(
        zip(zero_features, point_features), start=1
    ):
        difference = (point_feature - zero_feature).abs()
        spatial_change = difference.amax(dim=1)[0] > 0
        coordinates = torch.nonzero(spatial_change)

        print(f"Feature {index}:")
        if coordinates.numel() == 0:
            print("  no changed spatial positions")
            continue

        y_min, x_min = coordinates.amin(dim=0).tolist()
        y_max, x_max = coordinates.amax(dim=0).tolist()
        print(f"  y range = {y_min} ~ {y_max}")
        print(f"  x range = {x_min} ~ {x_max}")
