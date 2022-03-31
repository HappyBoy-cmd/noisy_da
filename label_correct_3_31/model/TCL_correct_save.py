from calendar import c
from os import pread
import torch.nn as nn
import model.backbone as backbone
from model.modules.grl import WarmStartGradientReverseLayer
import torch.nn.functional as F
import torch
import numpy as np
import random
import math


class GradientReverseLayer(torch.autograd.Function):

    def __init__(self,
                 iter_num=0,
                 alpha=1.0,
                 low_value=0.0,
                 high_value=0.1,
                 max_iter=1000.0):
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, grad_output):
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) /
            (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) -
            (self.high_value - self.low_value) + self.low_value)
        return -self.coeff * grad_output


class TCLNet(nn.Module):

    def __init__(self,
                 base_net='ResNet50',
                 use_bottleneck=True,
                 bottleneck_dim=256,
                 width=256,
                 class_num=31):
        super(TCLNet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.,
                                                       lo=0.,
                                                       hi=1.,
                                                       max_iters=1000,
                                                       auto_step=True)
        self.bottleneck_layer_list = [
            nn.Linear(self.base_network.output_num(), bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        ]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, class_num)
        ]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, 1)
        ]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)

        ## collect parameters, TCL does not train the base network.
        self.parameter_list = [{
            "params": self.bottleneck_layer.parameters(),
            "lr": 1
        }, {
            "params": self.classifier_layer.parameters(),
            "lr": 1
        }, {
            "params": self.classifier_layer_2.parameters(),
            "lr": 1
        }]

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.classifier_layer_2(features_adv)
        outputs_adv = self.sigmoid(outputs_adv)
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv


class TCL_correct_save(object):

    def __init__(self,
                 base_net='ResNet50',
                 width=1024,
                 class_num=31,
                 use_bottleneck=True,
                 use_gpu=True,
                 srcweight=3):
        self.c_net = TCLNet(base_net, use_bottleneck, width, width, class_num)
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight
        self.center_source = torch.zeros(class_num, width).cuda()
        self.center_target = torch.zeros(class_num, width).cuda()

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()
        class_criterion_pred = nn.CrossEntropyLoss(reduction='none')
        domain_criterion = nn.BCELoss()

        #### prediction
        if self.iter_num >= 2000:
            self.c_net.eval()
            _, outputs, softmax_outputs, _ = self.c_net(inputs)
            outputs = outputs.detach()
            softmax_outputs = softmax_outputs.detach()
            outputs_source = outputs.narrow(0, 0, labels_source.size(0))
            outputs_source_softmax = softmax_outputs.narrow(
                0, 0, labels_source.size(0))
            outputs_target_softmax = softmax_outputs.narrow(
                0, labels_source.size(0),
                inputs.size(0) - labels_source.size(0))
            classifier_loss = class_criterion_pred(outputs_source,
                                                   labels_source)
            classifier_loss = (classifier_loss - torch.min(classifier_loss)
                               ) / (torch.max(classifier_loss) -
                                    torch.min(classifier_loss))
            agey = 0.15  #-math.log(0.3)
            # weight_var = (classifier_loss < agey).int()
            weight_var = classifier_loss
            pred_hard = torch.max(outputs_source_softmax, dim=1)[1]

            self.c_net.train()

        #### loss compution
        if self.iter_num >= 2000:
            _, outputs, softmax_outputs, outputs_adv = self.c_net(inputs)

            outputs_source = outputs.narrow(0, 0, labels_source.size(0))

            outputs_source_softmax = softmax_outputs.narrow(
                0, 0, labels_source.size(0))
            outputs_target_softmax = softmax_outputs.narrow(
                0, labels_source.size(0),
                inputs.size(0) - labels_source.size(0))
        else:
            _, outputs, softmax_outputs, outputs_adv = self.c_net(inputs)

            outputs_source = outputs.narrow(0, 0, labels_source.size(0))

            outputs_source_softmax = softmax_outputs.narrow(
                0, 0, labels_source.size(0))
            outputs_target_softmax = softmax_outputs.narrow(
                0, labels_source.size(0),
                inputs.size(0) - labels_source.size(0))

        if self.iter_num < 2000:
            classifier_loss = class_criterion(outputs_source, labels_source)
            # en_loss = torch.zeros(1).cuda()
            en_loss = entropy(outputs_target_softmax)
            # Ld = torch.zeros(1).cuda()
            ## domain advcersarial loss
            source_domain_label = torch.FloatTensor(labels_source.size(0), 1)
            target_domain_label = torch.FloatTensor(
                inputs.size(0) - labels_source.size(0), 1)
            source_domain_label.fill_(1)
            target_domain_label.fill_(0)
            domain_label = torch.cat(
                [source_domain_label, target_domain_label], 0)
            domain_label = torch.autograd.Variable(domain_label.cuda())

            Ld = domain_criterion(outputs_adv, domain_label)
        else:
            beta = 0.8
            classifier_loss = cal_Ly(outputs_source, labels_source, pred_hard,
                                     weight_var, beta)
            en_loss = entropy(outputs_target_softmax)
            ## domain advcersarial loss
            source_domain_label = torch.FloatTensor(labels_source.size(0), 1)
            target_domain_label = torch.FloatTensor(
                inputs.size(0) - labels_source.size(0), 1)
            source_domain_label.fill_(1)
            target_domain_label.fill_(0)
            domain_label = torch.cat(
                [source_domain_label, target_domain_label], 0)
            domain_label = torch.autograd.Variable(domain_label.cuda())

            Ld = domain_criterion(outputs_adv, domain_label)

        self.iter_num += 1
        total_loss = classifier_loss + Ld + 0.1 * en_loss
        #print(classifier_loss.data, transfer_loss.data, en_loss.data)
        return [total_loss, classifier_loss, Ld]

    def predict(self, inputs):
        feature, outputs, softmax_outputs, _ = self.c_net(inputs)
        return softmax_outputs, feature, outputs

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode


def cal_center_loss(center_source, center_target):
    return torch.mean((torch.square(center_source - center_target)))


def get_center_source(feature,
                      label,
                      correct_feature,
                      given_label,
                      correct_label,
                      num_class=31):
    center = torch.zeros(num_class, feature.shape[1]).cuda()
    beta = 0.8
    for i in range(0, num_class):
        sum_fea = 0
        num_label = 0
        if sum(label == i) != 0:
            sum_fea += torch.sum(feature[label == i], 0)
            num_label = sum(label == i)
        if sum(given_label == i) != 0:
            sum_fea += 0.2 * (1 - beta) * torch.sum(
                correct_feature[given_label == i], 0)
            num_label += sum(given_label == i)
        if sum(correct_label == i) != 0:
            sum_fea += 0.2 * beta * torch.sum(
                correct_feature[correct_label == i], 0)
            num_label += sum(correct_label == i)
        if num_label > 0:
            center[i, :] = sum_fea / num_label
    return center


def get_center_target(feature, label, num_class=31):
    center = torch.zeros(num_class, feature.shape[1]).cuda()
    for i in range(0, num_class):
        if sum(label == i) != 0:
            center[i, :] = torch.sum(feature[label == i], 0) / sum(label == i)
    return center


def entropy(output_target):
    """
    entropy minimization loss on target domain data
    """
    output = output_target
    en = -torch.sum((output * torch.log(output + 1e-8)), 1)
    return torch.mean(en)


def entropy_select(output_target):
    """
    entropy minimization loss on target domain data
    """
    output = output_target
    en = -torch.sum((output * torch.log(output + 1e-8)), 1)
    return en


def linear_rampup(now_iter, total_iter=20000):
    if total_iter == 0:
        return 1.0
    else:
        current = np.clip(now_iter / total_iter, 0., 1.0)
        return float(current)


def cal_Ly(outputs_source, labels_source, pred_hard, weight_var, beta):

    # class_criterion = nn.CrossEntropyLoss()
    # if sum(weight_var == 1) > 0:
    #     loss_right = class_criterion(outputs_source[weight_var == 1],
    #                                  labels_source[weight_var == 1])
    # else:
    #     loss_right = 0
    # if sum(weight_var == 0) > 0:
    #     loss_correct_1 = (1 - beta) * class_criterion(
    #         outputs_source[weight_var == 0], labels_source[weight_var == 0])
    #     loss_correct_2 = beta * class_criterion(
    #         outputs_source[weight_var == 0], pred_hard[weight_var == 0])
    # else:
    #     loss_correct_1 = 0
    #     loss_correct_2 = 0
    # Ly = loss_right + 0.5 * (loss_correct_1 + loss_correct_2)
    # return Ly

    #######################################
    #######################################

    class_criterion = nn.CrossEntropyLoss(reduction='none')
    loss_correct_1 = torch.mean(
        (1 - weight_var) * class_criterion(outputs_source, labels_source))
    loss_correct_2 = torch.mean(weight_var *
                                class_criterion(outputs_source, pred_hard))
    Ly = (loss_correct_1 + loss_correct_2)
    return Ly
