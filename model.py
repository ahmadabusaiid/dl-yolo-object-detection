import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

architecture_config = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: tuples and then last integer represents number of repeats
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# for slides: something new adding batchnorm
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn_act=True, use_max_pool=False, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_bn_act = use_bn_act
        self.use_max_pool = use_max_pool
 
    def forward(self, x):
        if self.use_max_pool:
            return self.leakyrelu(self.batchnorm(self.conv(self.maxpool(x))))
        if self.use_bn_act:
            return self.leakyrelu(self.batchnorm(self.conv(x)))
        else:
            return self.conv(x)

# for slides: darknet
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fcs(**kwargs)
        self.initialize_weights()
    
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]
                
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                
            elif type(x) == list:
                conv1 = x[0] # Tuple
                conv2 = x[1] # Tuple
                num_repeats = x[2] # Integer
                
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    
                    in_channels = conv2[1]
                    
        return nn.Sequential(*layers)
    
    # for slide: original paper: 4096 
    def create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        
        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))
        # But we do a hack here
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 2048),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, S * S * (C + B * 5)), # (S, S, 30) where C + B * 5 = 30
        )
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    

#################################
#       Transfer Learning       #
#################################

class YOLOv1ResNet(nn.Module):
    def __init__(self, backbone_name = "resnet50", S=7, B=2, C=20):
        super().__init__()
        self.depth = B * 5 + C
        self.backbone_name = backbone_name
        self.S = S
        self.B = B
        self.C = C

        # Load backbone ResNet
        if backbone_name == "resnet18":
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone_name == "resnet34":
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif backbone_name == "resnet50":
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif backbone_name == "resnet101":
            backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif backbone_name == "resnet152":
            backbone = resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise ValueError("Invalid backbone model!")
        backbone.requires_grad_(False)            # Freeze backbone weights

        # Delete last two layers and attach detection layers
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()

        if backbone_name == "resnet18" or backbone_name == "resnet34":
            self.model = nn.Sequential(
            backbone,
            Reshape(512, 14, 14),
            DetectionNet(512, self.S, self.B, self.C)  # 4 conv, 2 linear
        )
        else:
            self.model = nn.Sequential(
                backbone,
                Reshape(2048, 14, 14),
                DetectionNet(2048, self.S, self.B, self.C)  # 4 conv, 2 linear
            )

    def forward(self, x):
        return self.model.forward(x)
    
class DetectionNet(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels, S, B, C):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        inner_channels = 1024
        self.depth = 5 * self.B + self.C
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1),   # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),

            nn.Linear(S * S * inner_channels, 2048),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(2048, self.S * self.S * self.depth)
        )
        self.initialize_weights()

    def forward(self, x):
        return torch.reshape(
            self.model.forward(x),
            (-1, self.S, self.S, self.depth)
        )
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class YOLO_V1_HeadV2_ResNet(nn.Module):
    def __init__(self, backbone_name = "resnet34", S=7, B=2, C=20):
        super().__init__()
        self.depth = B * 5 + C
        self.backbone_name = backbone_name
        self.S = S
        self.B = B
        self.C = C
 
        # Load backbone ResNet
        if backbone_name == "resnet18":
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone_name == "resnet34":
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            raise ValueError("Invalid backbone model!")
        backbone.requires_grad_(False)            # Freeze backbone weights
 
        # Delete last two layers and attach detection layers
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
 
        self.model = nn.Sequential(
            backbone,
            Reshape(512, 14, 14),
            YOLO_V2_HEAD(512, self.S, self.B, self.C)
        )
 
    def forward(self, x):
        return self.model.forward(x)
    
class YOLO_V2_HEAD(nn.Module):
    """The layers added on for detection as described in the paper."""
 
    def __init__(self, in_channels, S, B, C):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.depth = 5 * self.B + self.C
        self.in_channels = in_channels
        self.first_layer = self.create_first_layer()
        self.yv2_head = self.create_yv2_head()
        self.last_layer = self.create_last_layer()
        self.initialize_weights()
 
    def create_first_layer(self):
        layers = []
        layers += [
            CNNBlock(
                self.in_channels, out_channels=512, kernel_size=3, padding=1,
            )
        ]
        return nn.Sequential(*layers)
    
    def create_yv2_head(self):
        layers = []
        layers += [
            CNNBlock(512, 1024, kernel_size=3, padding=1, use_max_pool=True),
            CNNBlock(1024, 30, kernel_size=1),  # self.S * self.S * self.depth (normal way)
            CNNBlock(30, 1024, kernel_size=3, padding=1),
            CNNBlock(1024, 30, kernel_size=1),
            CNNBlock(30, 1024, kernel_size=3, padding=1),
        ]
        return nn.Sequential(*layers)
    
    def create_last_layer(self):
        layers = []
        layers += [
            CNNBlock(3072, 30, kernel_size=1, use_bn_act=False)
        ]
        return nn.Sequential(*layers)
 
    def forward(self, x):
        
        x1 = self.first_layer(x) # -> (-1, 512, 14, 14)
        
        chunk1 = x1[..., 0:7, 0:7]
        chunk2 = x1[..., 0:7, 7:14]
        chunk3 = x1[..., 7:14, 0:7]
        chunk4 = x1[..., 7:14, 7:14]
 
        x1_reshape = torch.cat((chunk1, chunk2, chunk3, chunk4), dim=1)
 
        x2 = self.yv2_head(x1) # -> (-1, 1024, 7, 7)
 
        x2_concat = torch.cat((x1_reshape, x2), dim=1) # -> (-1, 3072, 7, 7)
 
        x3 = self.last_layer(x2_concat)

        return torch.reshape(x3,
            (-1, self.S * self.S * self.depth)
        )
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


#############################
#       Helper Modules      #
#############################
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))