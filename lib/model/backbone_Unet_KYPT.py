import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet50_Weights

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type):
	
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
		       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
		       50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
		       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
		       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False) # RGB
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        skip_conn_layers = {} 
        in1 = x		   
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_conn_layers['stride2'] = x
        x = self.maxpool(x)
        x = self.layer1(x)
        skip_conn_layers['stride4'] = x
        x = self.layer2(x)
        skip_conn_layers['stride8'] = x
        x = self.layer3(x)
        skip_conn_layers['stride16'] = x
        x = self.layer4(x)
        skip_conn_layers['stride32'] = x

        return skip_conn_layers

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(ResNet50_Weights.IMAGENET1K_V2.url)
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        
        self.load_state_dict(org_resnet)
        print("Initialize resnet from model zoo")


class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.resnet = ResNetBackbone(50)
    
    def init_weights(self):
        self.resnet.init_weights()

    def forward(self, img):
        img_feat = self.resnet(img)
        return img_feat
    

class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        self.resnet_decoder = Decoder()

    def forward(self, img_feat, skip_conn_layers):
        feature_pyramid = self.resnet_decoder(img_feat, skip_conn_layers)
        return feature_pyramid

class DecoderNet_big(nn.Module):
    def __init__(self):
        super(DecoderNet_big, self).__init__()
        self.resnet_decoder = Decoder_big()

    def forward(self, img_feat, skip_conn_layers):
        feature_pyramid = self.resnet_decoder(img_feat, skip_conn_layers)
        return feature_pyramid

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        if resnet_type := 50:
            self.conv0d = make_conv_layers([2048, 512], kernel=1, padding=0)

            self.conv1d = make_conv_layers([1024, 256], kernel=1, padding=0)
            self.deconv1 = make_deconv_layers([2048, 256])
            self.conv1 = make_conv_layers([512, 256])

            self.conv2d = make_conv_layers([512, 128], kernel=1, padding=0)
            self.deconv2 = make_deconv_layers([256, 128])
            self.conv2 = make_conv_layers([256, 128])

            self.conv3d = make_conv_layers([256, 64], kernel=1, padding=0)
            self.deconv3 = make_deconv_layers([128, 64])
            self.conv3 = make_conv_layers([128, 64])

            self.conv4d = make_conv_layers([64, 32], kernel=1, padding=0)
            self.deconv4 = make_deconv_layers([64, 64])
            self.conv4 = make_conv_layers([64+32, 32])
        else:
            self.conv1d = make_conv_layers([256, 256], kernel=1, padding=0)
            self.deconv1 = make_deconv_layers([512, 256])
            self.conv1 = make_conv_layers([512, 256])

            self.conv2d = make_conv_layers([128, 128], kernel=1, padding=0)
            self.deconv2 = make_deconv_layers([256, 128])
            self.conv2 = make_conv_layers([256, 128])

            self.conv3d = make_conv_layers([64, 64], kernel=1, padding=0)
            self.deconv3 = make_deconv_layers([128, 64])
            self.conv3 = make_conv_layers([128, 64])

            self.conv4d = make_conv_layers([64, 32], kernel=1, padding=0)
            self.deconv4 = make_deconv_layers([64, 64])
            self.conv4 = make_conv_layers([64 + 32, 32])


    def forward(self, img_feat, skip_conn_layers):
        feature_pyramid = {}
        assert isinstance(skip_conn_layers, dict)
        if resnet_type := 50:
            feature_pyramid['stride32'] = self.conv0d(img_feat)
        else:
            feature_pyramid['stride32'] = img_feat

        skip_stride16_d = self.conv1d(skip_conn_layers['stride16']) #512
        deconv_img_feat1 = self.deconv1(img_feat)
        deconv_img_feat1_cat = torch.cat((skip_stride16_d, deconv_img_feat1),1)
        deconv_img_feat1_cat_conv = self.conv1(deconv_img_feat1_cat) #256
        feature_pyramid['stride16'] = deconv_img_feat1_cat_conv #256

        skip_stride8_d = self.conv2d(skip_conn_layers['stride8'])  # 256
        deconv_img_feat2 = self.deconv2(deconv_img_feat1_cat_conv)
        deconv_img_feat2_cat = torch.cat((skip_stride8_d, deconv_img_feat2), 1)
        deconv_img_feat2_cat_conv = self.conv2(deconv_img_feat2_cat) # 128
        feature_pyramid['stride8'] = deconv_img_feat2_cat_conv # 128

        skip_stride4_d = self.conv3d(skip_conn_layers['stride4'])  # 128
        deconv_img_feat3 = self.deconv3(deconv_img_feat2_cat_conv)
        deconv_img_feat3_cat = torch.cat((skip_stride4_d, deconv_img_feat3), 1)
        deconv_img_feat3_cat_conv = self.conv3(deconv_img_feat3_cat) # 64
        feature_pyramid['stride4'] = deconv_img_feat3_cat_conv # 64

        skip_stride2_d = self.conv4d(skip_conn_layers['stride2'])  # 32
        deconv_img_feat4 = self.deconv4(deconv_img_feat3_cat_conv)
        deconv_img_feat4_cat = torch.cat((skip_stride2_d, deconv_img_feat4), 1)
        deconv_img_feat4_cat_conv = self.conv4(deconv_img_feat4_cat) # 16x128x128 (featxhxw)
        feature_pyramid['stride2'] = deconv_img_feat4_cat_conv

        return feature_pyramid

class Decoder_big(nn.Module):
    def __init__(self):
        super(Decoder_big, self).__init__()
        self.deconv1 = make_deconv_layers([2048, 1024])
        self.conv1 = make_conv_layers([2048, 1024])

        self.deconv2 = make_deconv_layers([1024, 512])
        self.conv2 = make_conv_layers([1024, 512])

        self.deconv3 = make_deconv_layers([512, 256])
        self.conv3 = make_conv_layers([512, 256])

        self.deconv4 = make_deconv_layers([256, 128])
        self.conv4 = make_conv_layers([64+128, 128])


    def forward(self, img_feat, skip_conn_layers):
        feature_pyramid = {}
        assert isinstance(skip_conn_layers, dict)
        feature_pyramid['stride32'] = img_feat

        deconv_img_feat1 = self.deconv1(img_feat)
        deconv_img_feat1_cat = torch.cat((skip_conn_layers['stride16'], deconv_img_feat1),1)
        deconv_img_feat1_cat_conv = self.conv1(deconv_img_feat1_cat)
        feature_pyramid['stride16'] = deconv_img_feat1_cat_conv

        deconv_img_feat2 = self.deconv2(deconv_img_feat1_cat_conv)
        deconv_img_feat2_cat = torch.cat((skip_conn_layers['stride8'], deconv_img_feat2), 1)
        deconv_img_feat2_cat_conv = self.conv2(deconv_img_feat2_cat)
        feature_pyramid['stride8'] = deconv_img_feat2_cat_conv

        deconv_img_feat3 = self.deconv3(deconv_img_feat2_cat_conv)
        deconv_img_feat3_cat = torch.cat((skip_conn_layers['stride4'], deconv_img_feat3), 1)
        deconv_img_feat3_cat_conv = self.conv3(deconv_img_feat3_cat)
        feature_pyramid['stride4'] = deconv_img_feat3_cat_conv

        deconv_img_feat4 = self.deconv4(deconv_img_feat3_cat_conv)
        deconv_img_feat4_cat = torch.cat((skip_conn_layers['stride2'], deconv_img_feat4), 1)
        deconv_img_feat4_cat_conv = self.conv4(deconv_img_feat4_cat) # 128x128x128 (featxhxw)
        feature_pyramid['stride2'] = deconv_img_feat4_cat_conv

        return feature_pyramid
    

class Decoder_mid(nn.Module):
    def __init__(self):
        super(Decoder_big, self).__init__()
        self.conv0d = make_conv_layers([2048, 512], kernel=1, padding=0)

        self.conv1d = make_conv_layers([1024, 256], kernel=1, padding=0)
        self.deconv1 = make_deconv_layers([2048, 256])
        self.conv1 = make_conv_layers([512, 256])

        self.conv2d = make_conv_layers([512, 128], kernel=1, padding=0)
        self.deconv2 = make_deconv_layers([256, 128])
        self.conv2 = make_conv_layers([256, 128])

        self.conv3d = make_conv_layers([256, 64], kernel=1, padding=0)
        self.deconv3 = make_deconv_layers([128, 64])
        self.conv3 = make_conv_layers([128, 64])

        self.conv4d = make_conv_layers([64, 32], kernel=1, padding=0)
        self.deconv4 = make_deconv_layers([64, 64])
        self.conv4 = make_conv_layers([64+32, 32])


    def forward(self, img_feat, skip_conn_layers):
        feature_pyramid = {}
        assert isinstance(skip_conn_layers, dict)
        feature_pyramid['stride32'] = self.conv0d(img_feat)

        skip_stride16_d = self.conv1d(skip_conn_layers['stride16']) #512
        deconv_img_feat1 = self.deconv1(img_feat)
        deconv_img_feat1_cat = torch.cat((skip_stride16_d, deconv_img_feat1),1)
        deconv_img_feat1_cat_conv = self.conv1(deconv_img_feat1_cat) #256
        feature_pyramid['stride16'] = deconv_img_feat1_cat_conv #256

        skip_stride8_d = self.conv2d(skip_conn_layers['stride8'])  # 256
        deconv_img_feat2 = self.deconv2(deconv_img_feat1_cat_conv)
        deconv_img_feat2_cat = torch.cat((skip_stride8_d, deconv_img_feat2), 1)
        deconv_img_feat2_cat_conv = self.conv2(deconv_img_feat2_cat) # 128
        feature_pyramid['stride8'] = deconv_img_feat2_cat_conv # 128

        skip_stride4_d = self.conv3d(skip_conn_layers['stride4'])  # 128
        deconv_img_feat3 = self.deconv3(deconv_img_feat2_cat_conv)
        deconv_img_feat3_cat = torch.cat((skip_stride4_d, deconv_img_feat3), 1)
        deconv_img_feat3_cat_conv = self.conv3(deconv_img_feat3_cat) # 64
        feature_pyramid['stride4'] = deconv_img_feat3_cat_conv # 64

        skip_stride2_d = self.conv4d(skip_conn_layers['stride2'])  # 32
        deconv_img_feat4 = self.deconv4(deconv_img_feat3_cat_conv)
        deconv_img_feat4_cat = torch.cat((skip_stride2_d, deconv_img_feat4), 1)
        deconv_img_feat4_cat_conv = self.conv4(deconv_img_feat4_cat) # 16x128x128 (featxhxw)
        feature_pyramid['stride2'] = deconv_img_feat4_cat_conv

        return feature_pyramid


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.backbone = BackboneNet()
        self.decoder = Decoder_big()

    def init_weights(self):
        self.backbone.init_weights()

    def forward(self, img):
        enc_feat_dt = self.backbone(img)
        dec_feat_dt = self.decoder(enc_feat_dt['stride32'], enc_feat_dt)
        return enc_feat_dt, dec_feat_dt