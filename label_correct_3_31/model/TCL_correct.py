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
        self.source_classifier_layer_list = [
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, class_num)
        ]
        self.source_classifier = nn.Sequential(
            *self.source_classifier_layer_list)
        self.target_classifier_layer_list = [
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, class_num)
        ]
        self.target_classifier = nn.Sequential(
            *self.target_classifier_layer_list)
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
        self.parameter_list = [{
            "params": self.base_network.parameters(),
            "lr": 0.1
        }, {
            "params": self.bottleneck_layer.parameters(),
            "lr": 1
        }, {
            "params": self.source_classifier.parameters(),
            "lr": 1
        }, {
            "params": self.target_classifier.parameters(),
            "lr": 1
        }, {
            "params": self.domain_classifier.parameters(),
            "lr": 1
        }]

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.domain_classifier(features_adv)
        outputs_adv = self.sigmoid(outputs_adv)
        outputs = self.source_classifier(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv

    def get_target_classifier(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        outputs = self.target_classifier(features)
        softmax_outputs = self.softmax(outputs)

        return outputs, softmax_outputs


class TCL_correct(object):

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

    def get_loss(self, inputs_source, labels_source, inputs_target,
                 pseudo_labels_target, optimizer, bmm_model_source,bmm_model_maxLoss_source, bmm_model_minLoss_source, bmm_model_target, bmm_model_maxLoss_target, bmm_model_minLoss_target):
        class_criterion = nn.CrossEntropyLoss()
        class_criterion_pred = nn.CrossEntropyLoss(reduction='none')
        domain_criterion = nn.BCELoss()

        #### prediction
        optimizer.zero_grad()

        _, outputs_source, softmax_outputs_source, _ = self.c_net(
            inputs_source)
        outputs_source = outputs_source.detach()
        softmax_outputs_source = softmax_outputs_source.detach()

        outputs_target, softmax_outputs_target = self.c_net.get_target_classifier(
            inputs_target)
        outputs_target = outputs_target.detach()
        softmax_outputs_target = softmax_outputs_target.detach()

        optimizer.zero_grad()

        ####
        if self.iter_num >= 3000:
            B_s = compute_probabilities_batch(inputs_source, labels_source,
                                            self.c_net, bmm_model_source,
                                            bmm_model_maxLoss_source, bmm_model_minLoss_source)
            B_s = B_s.cuda()
            B_s[B_s <= 1e-4] = 1e-4
            B_s[B_s >= 1 - 1e-4] = 1 - 1e-4

            ####
            B_t = compute_probabilities_batch(inputs_target, pseudo_labels_target,
                                            self.c_net, bmm_model_target,
                                            bmm_model_maxLoss_target, bmm_model_minLoss_target)
            B_t = B_t.cuda()
            B_t[B_t <= 1e-4] = 1e-4
            B_t[B_t >= 1 - 1e-4] = 1 - 1e-4

        # outputs_source = F.log_softmax(outputs_source, dim=1)
        self_pred_source = torch.max(softmax_outputs_source, dim=1)[1]

        # outputs_target = F.log_softmax(outputs_target, dim=1)
        self_pred_target = torch.max(softmax_outputs_target, dim=1)[1]

        _, outputs_cls_source, _, outputs_adv_source = self.c_net(inputs_source)
        _, _, outputs_target_softmax, outputs_adv_target = self.c_net(inputs_target)
        outputs_adv = torch.cat((outputs_adv_source, outputs_adv_target), dim=0)
        outputs_cls_target, softmax_outputs_target = self.c_net.get_target_classifier(
            inputs_target)

        if self.iter_num < 2000:
            classifier_loss = class_criterion(outputs_cls_source, labels_source)
            cls_tar_loss = torch.zeros(1).cuda()
            en_loss = entropy(outputs_target_softmax)
            ## domain advcersarial loss
            source_domain_label = torch.FloatTensor(labels_source.size(0), 1)
            target_domain_label = torch.FloatTensor(pseudo_labels_target.size(0), 1)
            source_domain_label.fill_(1)
            target_domain_label.fill_(0)
            domain_label = torch.cat(
                [source_domain_label, target_domain_label], 0)
            domain_label = torch.autograd.Variable(domain_label.cuda())
            Ld = domain_criterion(outputs_adv, domain_label)

        elif self.iter_num <= 3000:
            classifier_loss = class_criterion(outputs_cls_source, labels_source)
            cls_tar_loss = class_criterion(outputs_cls_target,pseudo_labels_target)
            en_loss = entropy(outputs_target_softmax)
            ## domain advcersarial loss
            source_domain_label = torch.FloatTensor(labels_source.size(0), 1)
            target_domain_label = torch.FloatTensor(pseudo_labels_target.size(0), 1)
            source_domain_label.fill_(1)
            target_domain_label.fill_(0)
            domain_label = torch.cat(
                [source_domain_label, target_domain_label], 0)
            domain_label = torch.autograd.Variable(domain_label.cuda())
            Ld = domain_criterion(outputs_adv, domain_label)

        else:
            classifier_loss = cal_Ly_new(outputs_cls_source, labels_source, self_pred_source, B_s)
            cls_tar_loss = cal_Ly_new(outputs_cls_target, pseudo_labels_target, self_pred_target, B_t)
            en_loss = entropy(outputs_target_softmax)
            ## domain advcersarial loss
            source_domain_label = torch.FloatTensor(labels_source.size(0), 1)
            target_domain_label = torch.FloatTensor(pseudo_labels_target.size(0), 1)
            source_domain_label.fill_(1)
            target_domain_label.fill_(0)
            domain_label = torch.cat(
                [source_domain_label, target_domain_label], 0)
            domain_label = torch.autograd.Variable(domain_label.cuda())
            Ld = domain_criterion(outputs_adv, domain_label)

        self.iter_num += 1
        total_loss = classifier_loss + Ld + 0.1 * en_loss + 0.5 * cls_tar_loss
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


def compute_probabilities_batch(data, target, cnn_model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss):
    cnn_model.eval()
    _, outputs , _, _ = cnn_model(data)
    outputs = F.log_softmax(outputs, dim=1)
    batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
    batch_losses.detach_()
    outputs.detach_()
    cnn_model.train()
    batch_losses = (batch_losses - bmm_model_minLoss) / (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)
    batch_losses[batch_losses >= 1] = 1-10e-4
    batch_losses[batch_losses <= 0] = 10e-4

    #B = bmm_model.posterior(batch_losses,1)
    B = bmm_model.look_lookup(batch_losses, bmm_model_maxLoss, bmm_model_minLoss)

    return torch.FloatTensor(B)



def entropy(output_target):
    """
    entropy minimization loss on target domain data
    """
    output = output_target
    en = -torch.sum((output * torch.log(output + 1e-8)), 1)
    return torch.mean(en)




def linear_rampup(now_iter, total_iter=20000):
    if total_iter == 0:
        return 1.0
    else:
        current = np.clip(now_iter / total_iter, 0., 1.0)
        return float(current)




def cal_Ly_new(outputs, labels, pred, B):
    class_criterion = nn.CrossEntropyLoss(reduction='none')
    loss_x1_vec = (1 - B) * class_criterion(outputs, labels)
    loss_x1 = torch.sum(loss_x1_vec) / len(loss_x1_vec)
    loss_x1_pred_vec = B * class_criterion(outputs, pred)
    loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec) 
    loss = loss_x1 + loss_x1_pred
    
    return loss


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
