import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import os
from collections import OrderedDict


######################################################################
# Load parameters of model
# ---------------------------
def load_network(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'last')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load easy pretrained model: %s' % save_path)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Load model
# ---------------------------
def load_whole_network2(network, name, model_name=None, increment=True, d=5):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'whole_last')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load whole pretrained model: %s' % save_path)
    net_original = torch.load(save_path, weights_only=False)
    pretrained_dict = net_original.state_dict()
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    if increment:
        pretrained_dict = net_original.state_dict()
        inc_pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and pretrained_dict[k].shape != model_dict[k].shape}
        for k in inc_pretrained_dict.keys():
            if 'bias' in k:
                continue
            if model_dict[k].shape[0] >= inc_pretrained_dict[k].shape[0]:
                if 'did' in k:
                    s1 = int(model_dict[k].shape[0]/d)
                    s2 = int(inc_pretrained_dict[k].shape[0]/d)
                    for j in np.arange(d):
                        model_dict[k][s1*j: s1*j + s2] = \
                            inc_pretrained_dict[k][s2*j: s2*(j+1)]
                else:
                    model_dict[k][:inc_pretrained_dict[k].shape[0]] = inc_pretrained_dict[k]
    network.load_state_dict(model_dict)
    return network

def load_whole_network(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'whole_last')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load whole pretrained model: %s' % save_path)
    net_original = torch.load(save_path)
    pretrained_dict = net_original.state_dict()
    model_dict = network.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network


######################################################################
# Save parameters of model
# ---------------------------
def save_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.state_dict(), save_path)


######################################################################
# Save model
# ---------------------------
def save_whole_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network, save_path)


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_out')
        init.constant_(m.bias.detach(), 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.detach(), 1.0, 0.02)
        init.constant_(m.bias.detach(), 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.detach(), std=0.001)
        init.constant_(m.bias.detach(), 0.0)


######################################################################
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
# --------------------------------------------------------------------
class Fc_ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=False, num_bottleneck=512):
        super(Fc_ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f


class Fc(nn.Module):
    def __init__(self, input_dim, relu=False, output_dim=512):
        super(Fc, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, output_dim)]
        add_block += [nn.BatchNorm1d(output_dim)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.add_block = add_block

    def forward(self, x):
        x = self.add_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim=512, class_num=751, dropout=0.5):
        super(ClassBlock, self).__init__()
        classifier = []
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f

# Define a 2048 to 2 Model
class verif_net(nn.Module):
    def __init__(self):
        super(verif_net, self).__init__()
        self.classifier = Fc_ClassBlock(512, 2, dropout=0.75, relu=False)

    def forward(self, x):
        x = self.classifier.classifier(x)
        return x

# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, id_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # model_ft.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        # model_ft.layer4[0].conv2.stride = (1, 1)
        # model_ft.layer4[0].downsample[0].stride = (1, 1)
        self.model = model_ft
        self.fc = Fc(2048)
        self.id_classifier = ClassBlock(class_num=id_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        id_outputs, id_f = self.id_classifier(x)
        return id_outputs, id_f
######################################################################
# Define the DenseNet121-based Model
# --------------------------------------------------------------------
class ft_net_dense(nn.Module):
    def __init__(self, class_num=751, domain=3):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Fc(1024)
        self.classifier0 = ClassBlock(class_num=class_num)
        self.classifier1 = ClassBlock(class_num=class_num)
        self.classifier2 = ClassBlock(class_num=class_num)
        self.classifier3 = ClassBlock(class_num=class_num)
        self.classifier4 = ClassBlock(class_num=class_num)
        self.classifier5 = ClassBlock(class_num=class_num)
        self.classifier = [self.classifier0, self.classifier1, self.classifier2, self.classifier3, self.classifier4,
                           self.classifier5]
        self.domain = domain

    def forward(self, x):
        x = self.model.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        outputs = torch.FloatTensor().cuda()
        features = torch.FloatTensor().cuda()
        for d in range(self.domain):
            output, feature = self.classifier[d](x)
            outputs = torch.cat((outputs, output), 0)
            features = torch.cat((features, feature), 0)
        return outputs, features

class DisentangleNet(nn.Module):
    def __init__(self, did_embedding_net, sid_embedding_net):
        super(DisentangleNet, self).__init__()
        self.did_embedding_net = did_embedding_net
        self.sid_embedding_net = sid_embedding_net

    def forward(self, x):
        did_outputs, did_f = self.did_embedding_net(x)
        sid_outputs, sid_f = self.sid_embedding_net(x)
        return did_outputs, did_f, sid_outputs, sid_f

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.classifier = Fc_ClassBlock(1024, 2, dropout=0.5, relu=False)

    # def forward(self, x, train=False):
    #     output, feature, mid_coder, feature_coder = self.embedding_net(x)
    #     if train == False:
    #         return feature
    #     part_len = feature.shape[0] // 4
    #     feature_1_0 = feature[part_len * 0:part_len * 1]
    #     feature_1_1 = feature[part_len * 1:part_len * 2]
    #     feature_2_0 = feature[part_len * 2:part_len * 3]
    #     feature_2_1 = feature[part_len * 3:part_len * 4]
    #
    #     feature1 = (feature_1_0 - feature_2_0).pow(2)
    #     feature2 = (feature_1_1 - feature_2_1).pow(2)
    #     features = torch.cat((feature1, feature2), 0)
    #     results = self.classifier.classifier(features)
    #     part_len = results.shape[0] // 2
    #     result1 = results[part_len * 0:part_len * 1]
    #     result2 = results[part_len * 1:part_len * 2]
    #     return output, \
    #            result1, result2, \
    #            feature_1_0, feature_1_1, feature_2_0, feature_2_1

    def forward(self, x1_0, x1_1=None, x2_0=None, x2_1=None):
        output_1_0, feature_1_0, mid_coder_1_0, feature_coder_1_0 = self.embedding_net(x1_0)
        if x1_1 is None:
            return feature_1_0
        output_1_1, feature_1_1, mid_coder_1_1, feature_coder_1_1 = self.embedding_net(x1_1)
        output_2_0, feature_2_0, mid_coder_2_0, feature_coder_2_0 = self.embedding_net(x2_0)
        output_2_1, feature_2_1, mid_coder_2_1, feature_coder_2_1 = self.embedding_net(x2_1)
        feature1 = (feature_1_0 - feature_2_0).pow(2)
        result1 = self.classifier.classifier(feature1)
        feature2 = (feature_1_1 - feature_2_1).pow(2)
        result2 = self.classifier.classifier(feature2)
        return output_1_0, output_1_1, output_2_0, output_2_1, \
               result1, result2, \
               feature_1_0, feature_1_1, feature_2_0, feature_2_1



class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()

        self.part = 6  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, Fc_ClassBlock(2048, class_num, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        # sum prediction
        # y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


class PCB_test(nn.Module):
    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y
