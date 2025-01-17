from tqdm import tqdm
import numpy as np
import argparse
from torch.autograd import Variable
import torch
import sys
import os
import torch.nn as nn
# from model.TCL_correct import get_center_target

sys.path.insert(0, "/root/Messi_du/RDA")
from utils.config import Config


class INVScheduler(object):

    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter)**(-self.decay_rate)
        i = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i += 1
        return optimizer


#==============eval
def evaluate(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        probabilities, feature, _ = model_instance.predict(inputs)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        feature = feature.data.float()

        if first_test:
            all_probs = probabilities
            all_labels = labels
            all_feature = feature

            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)
            all_feature = torch.cat((all_feature, feature), 0)

    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_labels).float() / float(
            all_labels.size()[0])
    model_instance.set_train(ori_train_state)
    return {'accuracy': accuracy}, all_feature


### analysis
#==============eval
def save_loss(model_instance, input_loader, iter_num, dataset_name):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    class_criterion = nn.CrossEntropyLoss(reduction='none')

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels, labels_real = data[1][:, 0], data[1][:, 1]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        _, _, outputs = model_instance.predict(inputs)

        classifier_loss = class_criterion(outputs, labels)
        classifier_loss = classifier_loss.data.float()
        # labels = labels.data

        if first_test:
            all_classifier_loss = classifier_loss
            all_labels = labels
            all_labels_real = labels_real

            first_test = False
        else:
            all_classifier_loss = torch.cat(
                (all_classifier_loss, classifier_loss), 0)
            all_labels = torch.cat((all_labels, labels), 0)
            all_labels_real = torch.cat((all_labels_real, labels_real), 0)

    # _, predict = torch.max(all_probs, 1)
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).float() / float(all_labels.size()[0])

    model_instance.set_train(ori_train_state)

    # return {'accuracy':accuracy}, all_feature
    # save
    np.savez(
        os.path.join(
            '/root/Messi_du/RDA/plot', 'TCL_loss_only_classification' +
            dataset_name + '_' + str(iter_num) + '_plot'),
        all_classifier_loss.cpu().numpy(),
        all_labels.cpu().numpy(),
        all_labels_real.cpu().numpy())


def get_center_source(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    class_criterion = nn.CrossEntropyLoss(reduction='none')

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1][:, 0]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        with torch.no_grad():
            _, fea, outputs = model_instance.predict(inputs)

        classifier_loss = class_criterion(outputs, labels)
        classifier_loss = (classifier_loss - torch.min(classifier_loss)) / (
            torch.max(classifier_loss) - torch.min(classifier_loss))
        # classifier_loss = classifier_loss.data.float()
        agey = 0.15  #-math.log(0.3)
        weight_var = (classifier_loss < agey).int()
        # labels = labels.data

        if first_test:
            all_feas = fea[weight_var == 1].cpu()
            all_labels = labels[weight_var == 1].cpu()
            first_test = False
        else:
            all_labels = torch.cat((all_labels, labels[weight_var == 1].cpu()),
                                   0)
            all_feas = torch.cat((all_feas, fea[weight_var == 1].cpu()), 0)
        del fea
        del outputs
        torch.cuda.empty_cache()
    center = get_center_all(all_feas, all_labels, model_instance.class_num)
    model_instance.set_train(ori_train_state)
    return center


def get_center_target(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    class_criterion = nn.CrossEntropyLoss(reduction='none')

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        # labels = data[1]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            # labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            # labels = Variable(labels)
        with torch.no_grad():
            softmax_outputs, fea, _ = model_instance.predict(inputs)
            pred = torch.max(softmax_outputs, dim=1)[1]
        if first_test:
            all_feas = fea
            all_labels = pred
            first_test = False
        else:
            all_labels = torch.cat((all_labels, pred), 0)
            all_feas = torch.cat((all_feas, fea), 0)
        del fea
        del softmax_outputs
        torch.cuda.empty_cache()
    center = get_center_all(all_feas, all_labels, model_instance.class_num)
    model_instance.set_train(ori_train_state)
    return center


def get_center_all(feature, label, num_class=31):
    center = torch.zeros(num_class, feature.shape[1]).cuda()
    for i in range(0, num_class):
        if sum(label == i) != 0:
            center[i, :] = torch.sum(feature[label == i], 0) / sum(label == i)
    return center


##
def train(model_instance, train_source_noisy_loader, train_target_loader,
          test_target_loader, group_ratios, max_iter, optimizer, lr_scheduler,
          eval_interval, dataset_name):
    model_instance.set_train(True)
    print("start train...")
    loss = []  #accumulate total loss for visulization.
    result = []  #accumulate eval result on target data during training.
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm(desc='Train iter', total=max_iter)
    while True:
        for (datas_noisy,
             datat) in tqdm(zip(train_source_noisy_loader,
                                train_target_loader),
                            total=min(len(train_source_noisy_loader),
                                      len(train_target_loader)),
                            desc='Train epoch = {}'.format(epoch),
                            ncols=80,
                            leave=False):
            inputs_source, labels_source, _ = datas_noisy
            labels_source = labels_source[:, 0]
            # labels_source_real = labels_source[:, 1]
            # inputs_source_noisy, _, _ = datas_noisy
            inputs_target, labels_target, _ = datat

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer,
                                                    iter_num / 5)
            optimizer.zero_grad()

            if model_instance.use_gpu:
                inputs_source, inputs_target, labels_source = Variable(
                    inputs_source).cuda(), Variable(
                        inputs_target).cuda(), Variable(labels_source).cuda(),
            else:
                inputs_source, inputs_target, labels_source = Variable(
                    inputs_source), Variable(inputs_target), Variable(
                        labels_source),

            total_loss = train_batch(model_instance, inputs_source,
                                     labels_source, inputs_target, optimizer,
                                     iter_num, max_iter)

            #val
            if iter_num == 1990:
                center_source = get_center_source(model_instance,
                                                  train_source_noisy_loader)
                center_target = get_center_target(model_instance,
                                                  train_target_loader)
                model_instance.center_source = center_source.cuda()  ### 包含0
                model_instance.center_target = center_target.cuda()

            if iter_num % 400 == 0:
                print("total_loss:", total_loss[0], "classify loss:",
                      total_loss[1], "loss_d", total_loss[2])
            if iter_num % eval_interval == 0 and iter_num != 0:
                eval_result, all_feature = evaluate(model_instance,
                                                    test_target_loader)
                save_loss(model_instance, train_source_noisy_loader, iter_num,
                          dataset_name)
                print('source domain:', eval_result)
                result.append(eval_result['accuracy'].cpu().data.numpy())

            iter_num += 1
            total_progress_bar.update(1)
            loss.append(total_loss)

        epoch += 1

        if iter_num > max_iter:
            #np.save('statistic/TCL_feature_target.npy', all_feature.cpu().numpy())
            break
    print('finish train')
    #torch.save(model_instance.c_net.state_dict(), 'statistic/TCL_model.pth')
    return [loss, result]


def train_batch(model_instance, inputs_source, labels_source, inputs_target,
                optimizer, iter_num, max_iter):
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    total_loss = model_instance.get_loss(inputs, labels_source)
    total_loss[0].backward()
    optimizer.step()
    return [
        total_loss[0].cpu().data.numpy(), total_loss[1].cpu().data.numpy(),
        total_loss[2].cpu().data.numpy()
    ]


if __name__ == '__main__':
    from model.TCL_correct import TCL_correct
    from preprocess.data_provider import load_images_TCL
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        help='all sets of configuration parameters',
                        default='/root/Messi_du/RDA/config/dann.yml')
    parser.add_argument('--dataset',
                        default='Office-31',
                        type=str,
                        help='which dataset')
    parser.add_argument('--gpu_id',
                        default='1',
                        type=str,
                        help='which dataset')
    parser.add_argument(
        '--src_address',
        default=
        "/root/Messi_du/RDA/data/Office-31/webcam_uniform_noisy_0.4.txt",
        type=str,
        help='address of image list of source dataset')
    parser.add_argument('--tgt_address',
                        default="/root/Messi_du/RDA/data/Office-31/amazon.txt",
                        type=str,
                        help='address of image list of target dataset')
    parser.add_argument(
        '--stats_file',
        default=
        "/root/Messi_du/RDA/statistic/TCL-W2A-uniform-noisy-0.4-2022-02-22-13-37-40.pkl",
        type=str,
        help='store the training loss and and validation acc')
    parser.add_argument('--noisy_rate',
                        default=None,
                        type=float,
                        help='noisy rate')
    args = parser.parse_args()

    cfg = Config(args.config)
    print(args)
    source_file = args.src_address
    target_file = args.tgt_address
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.dataset == 'Office-31':
        class_num = 31
        width = 256
        srcweight = 4
        is_cen = False
    elif args.dataset == 'Office-home':
        class_num = 65
        width = 256
        srcweight = 2
        is_cen = False
    elif args.dataset == 'COVID-19':
        class_num = 3
        width = 256
        srcweight = 4
        is_cen = False
        # Another choice for Office-home:
        # width = 1024
        # srcweight = 3
        # is_cen = True
    else:
        width = -1

    model_instance = TCL_correct(base_net='ResNet50',
                                 width=width,
                                 use_gpu=True,
                                 class_num=class_num,
                                 srcweight=srcweight)

    train_source_noisy_loader = load_images_TCL(source_file,
                                                batch_size=32,
                                                is_cen=is_cen,
                                                split_noisy=False)
    # train_source_noisy_loader = train_source_clean_loader
    train_target_loader = load_images_TCL(target_file,
                                          batch_size=32,
                                          is_cen=is_cen)
    test_target_loader = load_images_TCL(target_file,
                                         batch_size=32,
                                         is_train=False)

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'

    optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)

    assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=cfg.init_lr)
    to_dump = train(model_instance,
                    train_source_noisy_loader,
                    train_target_loader,
                    test_target_loader,
                    group_ratios,
                    max_iter=18000,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    eval_interval=500,
                    dataset_name=args.dataset)
    pickle.dump(to_dump, open(args.stats_file, 'wb'))
