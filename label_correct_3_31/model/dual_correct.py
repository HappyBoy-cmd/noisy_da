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


class dual_Net(nn.Module):

    def __init__(self,
                 base_net='ResNet50',
                 use_bottleneck=True,
                 bottleneck_dim=256,
                 width=256,
                 class_num=31):
        super(dual_Net, self).__init__()
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
        self.source_classifier_layer = [
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, class_num)
        ]
        self.source_classifier = nn.Sequential(*self.source_classifier_layer)

        ##
        self.target_classifier_layer = [
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, class_num)
        ]
        self.target_classifier = nn.Sequential(*self.target_classifier_layer)

        ##
        self.domain_classifier_list = [
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, 1)
        ]
        self.domain_classifier = nn.Sequential(*self.domain_classifier_list)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.domain_classifier[dep * 3].weight.data.normal_(0, 0.01)
            self.domain_classifier[dep * 3].bias.data.fill_(0.0)
            self.source_classifier[dep * 3].weight.data.normal_(0, 0.01)
            self.source_classifier[dep * 3].bias.data.fill_(0.0)
            self.target_classifier[dep * 3].weight.data.normal_(0, 0.01)
            self.target_classifier[dep * 3].bias.data.fill_(0.0)

        ## collect parameters, TCL does not train the base network.
        self.parameter_list = [
            {
                "params": self.bottleneck_layer.parameters(),
                "lr": 1
            },
            {
                "params": self.source_classifier.parameters(),
                "lr": 1
            },
            {
                "params": self.domain_classifier.parameters(),
                "lr": 1
            },
            {
                "params": self.target_classifier.parameters(),
                "lr": 1
            },
        ]

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.domain_classifier(features_adv)
        outputs_adv = self.sigmoid(outputs_adv)
        outputs_by_sf = self.source_classifier(features)
        outputs_by_tf = self.target_classifier(features)
        softmax_outputs_by_sf = self.softmax(outputs_by_sf)
        softmax_outputs_by_tf = self.softmax(outputs_by_tf)

        return features, outputs_by_sf, softmax_outputs_by_sf, outputs_adv, outputs_by_tf, softmax_outputs_by_tf


class dual_correct(object):

    def __init__(self,
                 base_net='ResNet50',
                 width=1024,
                 class_num=31,
                 use_bottleneck=True,
                 use_gpu=True,
                 srcweight=3):
        self.c_net = dual_Net(base_net, use_bottleneck, width, width,
                              class_num)
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
        if self.iter_num > 1000 and self.iter_num < 2000:
            self.c_net.eval()

            features, outputs_by_sf, softmax_outputs_by_sf, _, _, _ = self.c_net(
                inputs)
            outputs_by_sf = outputs_by_sf.detach()
            softmax_outputs_by_sf = softmax_outputs_by_sf.detach()
            outputs_target_softmax_by_sf = softmax_outputs_by_sf.narrow(
                0, labels_source.size(0),
                inputs.size(0) - labels_source.size(0))
            pred_hard_target_by_sf = torch.max(outputs_target_softmax_by_sf,
                                               dim=1)[1]

            self.c_net.train()

        if self.iter_num >= 2000:
            self.c_net.eval()
            features, outputs_by_sf, softmax_outputs_by_sf, _, outputs_by_tf, softmax_outputs_by_tf = self.c_net(
                inputs)
            outputs_by_sf = outputs_by_sf.detach()
            softmax_outputs_by_sf = softmax_outputs_by_sf.detach()
            outputs_source_by_sf = outputs_by_sf.narrow(
                0, 0, labels_source.size(0))
            outputs_source_softmax_by_sf = softmax_outputs_by_sf.narrow(
                0, 0, labels_source.size(0))
            outputs_target_softmax_by_sf = softmax_outputs_by_sf.narrow(
                0, labels_source.size(0),
                inputs.size(0) - labels_source.size(0))
            outputs_target_by_tf = outputs_by_tf.narrow(
                0, labels_source.size(0),
                inputs.size(0) - labels_source.size(0))
            outputs_target_softmax_by_tf = softmax_outputs_by_tf.narrow(
                0, labels_source.size(0),
                inputs.size(0) - labels_source.size(0))

            ### select_source_sample
            classifier_loss = class_criterion_pred(outputs_source_by_sf,
                                                   labels_source)
            classifier_loss = (classifier_loss - torch.min(classifier_loss)
                               ) / (torch.max(classifier_loss) -
                                    torch.min(classifier_loss))
            agey = 0.15  #-math.log(0.3)
            weight_var = (classifier_loss < agey).int()
            pred_hard_source_by_sf = torch.max(outputs_source_softmax_by_sf,
                                               dim=1)[1]

            ### select_target_sample
            pred_hard_target_by_sf = torch.max(outputs_target_softmax_by_sf,
                                               dim=1)[1]
            classifier_loss_tar_by_tf = class_criterion_pred(
                outputs_target_by_tf, pred_hard_target_by_sf)
            classifier_loss_tar_by_tf = (
                classifier_loss_tar_by_tf -
                torch.min(classifier_loss_tar_by_tf)) / (
                    torch.max(classifier_loss_tar_by_tf) -
                    torch.min(classifier_loss_tar_by_tf))
            agey = 0.15  #-math.log(0.3)
            weight_var_target = (classifier_loss_tar_by_tf < agey).int()
            pred_hard_target_by_tf = torch.max(outputs_target_softmax_by_tf,
                                               dim=1)[1]
            # en_loss_target = entropy_select(
            #     outputs_target_softmax_by_sf) / math.log(self.class_num)
            # aget = 0.9
            # weight_var_target = (en_loss_target < aget).int()

            #### update center
            features_source = features.narrow(0, 0, labels_source.size(0))
            features_target = features.narrow(
                0, labels_source.size(0),
                inputs.size(0) - labels_source.size(0))
            select_features = features_source[weight_var == 1]
            select_label = labels_source[weight_var == 1]
            # correct_feature = features_source[weight_var == 0]
            # correct_label = pred_hard_source_by_sf[weight_var == 0]
            # given_label = labels_source[weight_var == 0]
            # current_center_source = get_center_source(select_features,
            #                                           select_label,
            #                                           correct_feature,
            #                                           given_label,
            #                                           correct_label)
            current_center_source = get_center(
                select_features,
                select_label,
            )
            current_center_target = get_center(
                features_target[weight_var_target == 1],
                pred_hard_target_by_sf[weight_var_target ==
                                       1])  ### need to change

            alpha = 0.3
            for i in range(0, self.class_num):
                if sum(select_label == i) > 0:
                    center_source = alpha * current_center_source + (
                        1 - alpha) * self.center_source
                if sum(pred_hard_target_by_sf[weight_var_target == 1] ==
                       i) > 0:
                    center_target = alpha * current_center_target + (
                        1 - alpha) * self.center_target

            self.center_source = center_source.detach()
            self.center_target = center_target.detach()

            ### center label
            pred_source_hard_center = pred_center(features_source,
                                                  center_target)

            pred_target_hard_center = pred_center(features_target,
                                                  center_source)

            ###

            ###
            self.c_net.train()

        #### loss compution
        if self.iter_num > 1000:

            _, outputs_by_sf, softmax_outputs_by_sf, outputs_adv, outputs_by_tf, _ = self.c_net(
                inputs)

            outputs_source_by_sf = outputs_by_sf.narrow(
                0, 0, labels_source.size(0))

            outputs_source_softmax_by_sf = softmax_outputs_by_sf.narrow(
                0, 0, labels_source.size(0))
            outputs_target_softmax_by_sf = softmax_outputs_by_sf.narrow(
                0, labels_source.size(0),
                inputs.size(0) - labels_source.size(0))

            outputs_target_by_tf = outputs_by_tf.narrow(
                0, labels_source.size(0),
                inputs.size(0) - labels_source.size(0))

        else:
            _, outputs_by_sf, softmax_outputs_by_sf, outputs_adv, outputs_by_tf, _ = self.c_net(
                inputs)

            outputs_source_by_sf = outputs_by_sf.narrow(
                0, 0, labels_source.size(0))

            outputs_source_softmax_by_sf = softmax_outputs_by_sf.narrow(
                0, 0, labels_source.size(0))
            outputs_target_softmax_by_sf = softmax_outputs_by_sf.narrow(
                0, labels_source.size(0),
                inputs.size(0) - labels_source.size(0))
            # outputs_target_by_target = outputs_by_target.narrow(
            #     0, labels_source.size(0),
            #     inputs.size(0) - labels_source.size(0))
        # d_source = outputs_adv.narrow(0, 0, labels_source.size(0))

        if self.iter_num <= 1000:
            classifier_loss = class_criterion(outputs_source_by_sf,
                                              labels_source)
            classifier_by_target_loss = torch.zeros(1).cuda()
            en_loss = torch.zeros(1).cuda()
            # en_loss = entropy(outputs_target_softmax_by_sf)
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
        elif self.iter_num < 2000:
            classifier_loss = class_criterion(outputs_source_by_sf,
                                              labels_source)
            classifier_by_target_loss = class_criterion(
                outputs_target_by_tf, pred_hard_target_by_sf)
            en_loss = torch.zeros(1).cuda()
            # en_loss = entropy(outputs_target_softmax_by_sf)
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
            classifier_loss = cal_Ly(outputs_source_by_sf, labels_source, pred_hard_source_by_sf,
                                     weight_var, beta)
            classifier_by_target_loss = cal_Ly(outputs_target_by_tf, pred_hard_target_by_sf, pred_hard_target_by_tf,
                                     weight_var_target, beta)
            # classiifer_by_target_loss = class_criterion(
            #     outputs_target_by_tf[weight_var_target == 1],
            #     pred_hard_target_by_sf[weight_var_target == 1])
            # classifier_by_target_loss = cal_Ly_adap(outputs_target_by_tf,
            #                                         pred_hard_target_by_sf,
            #                                         pred_hard_target_by_tf,
            #                                         pred_target_hard_center,
            #                                         weight_var_target, beta)
            # classifier_loss = cal_Ly_adap(outputs_source_by_sf, labels_source,
            #                               pred_hard_source_by_sf,
            #                               pred_source_hard_center, weight_var,
            #                               beta)
            en_loss = entropy(outputs_target_softmax_by_sf)
            # center_loss = cal_center_loss(center_source, center_target)

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
        total_loss = classifier_loss + Ld + 0.1 * en_loss + 0.5 * classifier_by_target_loss
        return [total_loss, classifier_loss, Ld]

    def predict(self, inputs):
        feature, outputs_by_sf, softmax_outputs_by_sf, _, outputs_by_tf, softmax_outputs_by_tf = self.c_net(
            inputs)
        return softmax_outputs_by_sf, feature, outputs_by_sf, softmax_outputs_by_tf

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode


def pred_center(features_source, center_target):
    pred = torch.zeros(len(features_source)).cuda()
    # num = 0
    # for fea in features_source:
    #     sim_list = []
    #     for j in range(len(center_target)):
    #         sim_list.append(cos_distance(fea, center_target[j, :]))
    #     max_sim = max(sim_list)
    #     pred[num] = sim_list.index(max_sim)
    #     num = num + 1
    dists = euclidean_dist(features_source, center_target)
    pred = torch.min(dists, dim=1)[1]
    return pred


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def cos_distance(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB)**0.5)


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


def get_center(feature, label, num_class=31):
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

    # agey = - math.log(0.3)
    # aged = - math.log(1.0 - 0.5)
    # age = agey + 0.1 * aged
    # y_softmax = source_y_softmax

    # the_index = torch.LongTensor(np.array(range(source_d.size(0)))).cuda()
    # y_label = y_softmax[the_index, label]
    # y_loss = - torch.log(y_label)

    # d_loss = - torch.log(1.0 - source_d)
    # d_loss = d_loss.view(source_d.size(0))

    # weight_loss = y_loss + 0.1 * d_loss

    # weight_var = (weight_loss < age).float().detach()
    class_criterion = nn.CrossEntropyLoss()
    if sum(weight_var == 1) > 0:
        loss_right = class_criterion(outputs_source[weight_var == 1],
                                     labels_source[weight_var == 1])
    else:
        loss_right = torch.zeros(1).cuda()
    if sum(weight_var == 0) > 0:
        loss_correct_1 = (1 - beta) * class_criterion(
            outputs_source[weight_var == 0], labels_source[weight_var == 0])
        loss_correct_2 = beta * class_criterion(
            outputs_source[weight_var == 0], pred_hard[weight_var == 0])
    else:
        loss_correct_1 = torch.zeros(1).cuda()
        loss_correct_2 = torch.zeros(1).cuda()
    Ly = loss_right + 0.5 * (loss_correct_1 + loss_correct_2)
    return Ly


def cal_Ly_adap(outputs_source, labels_observe, pred_hard, pred_hard_center,
                weight_var, beta):

    class_criterion = nn.CrossEntropyLoss()
    select_output_source = outputs_source[weight_var == 0]
    select_labels_observe = labels_observe[weight_var == 0]
    select_pred_hard = pred_hard[weight_var == 0]
    select_pred_hard_center = pred_hard_center[weight_var == 0]

    correct_output = select_output_source[select_pred_hard ==
                                          select_pred_hard_center]
    correct_labels_observe = select_labels_observe[select_pred_hard ==
                                                   select_pred_hard_center]
    correct_pred = select_pred_hard[select_pred_hard ==
                                    select_pred_hard_center]
    if sum(weight_var == 1) > 0:
        loss_right = class_criterion(outputs_source[weight_var == 1],
                                     labels_observe[weight_var == 1])
    else:
        loss_right = torch.zeros(1).cuda()
    if sum(weight_var == 0) > 0 and sum(
            select_pred_hard == select_pred_hard_center) > 0:
        loss_correct_1 = (1 - beta) * class_criterion(correct_output,
                                                      correct_labels_observe)
        loss_correct_2 = beta * class_criterion(correct_output, correct_pred)
    else:
        loss_correct_1 = torch.zeros(1).cuda()
        loss_correct_2 = torch.zeros(1).cuda()
    Ly = loss_right + 0.5 * (loss_correct_1 + loss_correct_2)
    return Ly
