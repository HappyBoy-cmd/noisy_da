from xml import dom
import tqdm
import numpy as np
import argparse
from torch.autograd import Variable
import torch
import sys
import os
import random
import torch.nn.functional as F
import scipy.stats as stats
from matplotlib import pyplot as plt

sys.path.insert(0, "/home/dyt/Messi_du/RDA_copy")
from utils.config import Config
from data.utils import build_dataset


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

def update_pseudo_labels_target(model_instance, data_loader, target_loader_track):
    model_instance.c_net.eval()
    updated_labels = np.zeros(len(data_loader.dataset))
    ###
    with torch.no_grad():
        for batch_idx, (inputs, _, _, index) in enumerate(data_loader):
            inputs = Variable(inputs).cuda()
            probabilities, feature, _ = model_instance.predict(inputs)
            pred = probabilities.data.max(1, keepdim=True)[1]
            updated_labels[index] = pred[:,0].cpu().numpy()
    data_loader.dataset.noisy_targets = updated_labels.astype(int).tolist()
    target_loader_track.dataset.noisy_targets = updated_labels.astype(int).tolist()
    model_instance.c_net.train()
    
def train(model_instance, train_source_noisy_loader,
          train_target_loader, test_target_loader, source_loader_track,
                    target_loader_track, group_ratios, max_iter,
          optimizer, lr_scheduler, eval_interval, bmm_model_source = 0,bmm_model_maxLoss_source =0, bmm_model_minLoss_source =0,
          bmm_model_target = 0,bmm_model_maxLoss_target =0, bmm_model_minLoss_target =0):
    model_instance.set_train(True)
    print("start train...")
    loss = []  #accumulate total loss for visulization.
    result = []  #accumulate eval result on target data during training.
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)
    while True:
        for (datas_noisy, datat) in tqdm.tqdm(zip(train_source_noisy_loader,
                                     train_target_loader),
                                 total=min(len(train_source_noisy_loader),
                                           len(train_target_loader)),
                                 desc='Train epoch = {}'.format(epoch),
                                 ncols=80,
                                 leave=False):
            inputs_source, _, labels_source, _ = datas_noisy
            inputs_target, _, pseudo_labels_target, _ = datat

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer,
                                                    iter_num / 5)
            optimizer.zero_grad()

            if model_instance.use_gpu:
                inputs_source, inputs_target, labels_source, pseudo_labels_target = Variable(
                    inputs_source).cuda(), Variable(
                        inputs_target).cuda(), Variable(labels_source).cuda(), Variable(pseudo_labels_target).cuda(),
            else:
                inputs_source, inputs_target, labels_source = Variable(
                    inputs_source), Variable(inputs_target), Variable(
                        labels_source)

            total_loss = train_batch(model_instance, inputs_source,
                                     labels_source, inputs_target, pseudo_labels_target, optimizer,
                                     iter_num, max_iter,bmm_model_source,bmm_model_maxLoss_source, bmm_model_minLoss_target, bmm_model_target, bmm_model_maxLoss_target, bmm_model_minLoss_target)

            #val
            if iter_num % eval_interval == 0 and iter_num != 0:
                eval_result_source, _ = evaluate(model_instance,
                                                 train_source_noisy_loader)
                print('source domain:', eval_result_source)
                eval_result, all_feature = evaluate(model_instance,
                                                    test_target_loader)
                print('target domain:', eval_result)
                result.append(eval_result['accuracy'].cpu().data.numpy())
                
                update_pseudo_labels_target(model_instance,train_target_loader, target_loader_track)
                
                _, _, _, bmm_model_source, bmm_model_maxLoss_source, bmm_model_minLoss_source = \
                track_training_loss(model_instance.c_net, source_loader_track, int(iter_num / eval_interval), bmm_model_source, bmm_model_maxLoss_source, bmm_model_minLoss_source, domain = 'source')
                
                _, _, _, bmm_model_target, bmm_model_maxLoss_target, bmm_model_minLoss_target = \
                track_training_loss(model_instance.c_net, target_loader_track, int(iter_num / eval_interval), bmm_model_target, bmm_model_maxLoss_target, bmm_model_minLoss_target, domain = 'target')

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




def track_training_loss(model, train_loader, epoch, bmm_model1, bmm_model_maxLoss1, bmm_model_minLoss1, domain = 'source'):
    model.eval()

    all_losses = torch.Tensor()
    all_predictions = torch.Tensor()
    all_probs = torch.Tensor()
    all_argmaxXentropy = torch.Tensor()

    for batch_idx, (data, _, target, _) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        if domain == 'source':
            _, prediction, _, _ = model(data)
        else:
            prediction, _ = model.get_target_classifier(data)

        prediction = F.log_softmax(prediction, dim=1)
        idx_loss = F.nll_loss(prediction, target, reduction = 'none')
        idx_loss.detach_()
        all_losses = torch.cat((all_losses, idx_loss.cpu()))
        probs = prediction.clone()
        probs.detach_()
        all_probs = torch.cat((all_probs, probs.cpu()))
        arg_entr = torch.max(prediction, dim=1)[1]
        arg_entr = F.nll_loss(prediction.float(), arg_entr.cuda(), reduction='none')
        arg_entr.detach_()
        all_argmaxXentropy = torch.cat((all_argmaxXentropy, arg_entr.cpu()))

    loss_tr = all_losses.data.numpy()

    # outliers detection
    max_perc = np.percentile(loss_tr, 95)
    min_perc = np.percentile(loss_tr, 5)
    loss_tr = loss_tr[(loss_tr<=max_perc) & (loss_tr>=min_perc)]

    bmm_model_maxLoss = torch.FloatTensor([max_perc]).cuda()
    bmm_model_minLoss = torch.FloatTensor([min_perc]).cuda() + 10e-6


    loss_tr = (loss_tr - bmm_model_minLoss.data.cpu().numpy()) / (bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6)

    loss_tr[loss_tr>=1] = 1-10e-4
    loss_tr[loss_tr <= 0] = 10e-4

    bmm_model = BetaMixture1D(max_iters=10)
    bmm_model.fit(loss_tr)

    bmm_model.create_lookup(1)
    
    model.train()

    return all_losses.data.numpy(), \
           all_probs.data.numpy(), \
           all_argmaxXentropy.numpy(), \
           bmm_model, bmm_model_maxLoss, bmm_model_minLoss
##############################################################################
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta



class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)





def train_batch(model_instance, inputs_source, labels_source, inputs_target, pseudo_labels_target, 
                optimizer, iter_num, max_iter,bmm_model_source,bmm_model_maxLoss_source, bmm_model_minLoss_source, bmm_model_target,bmm_model_maxLoss_target, bmm_model_minLoss_target):
    # inputs = torch.cat((inputs_source, inputs_target), dim=0)
    total_loss = model_instance.get_loss(inputs_source,  labels_source, inputs_target, pseudo_labels_target, optimizer,bmm_model_source,bmm_model_maxLoss_source, bmm_model_minLoss_source, bmm_model_target, bmm_model_maxLoss_target, bmm_model_minLoss_target)
    total_loss[0].backward()
    optimizer.step()
    return [
        total_loss[0].cpu().data.numpy(), total_loss[1].cpu().data.numpy(),
        total_loss[2].cpu().data.numpy()
    ]


if __name__ == '__main__':
    from model.TCL_correct import TCL_correct
    # from preprocess.data_provider import load_images
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='all sets of configuration parameters',
        default='/home/dyt/Messi_du/RDA_copy/config/dann.yml')
    parser.add_argument('--dataset',
                        default='Office-31',
                        type=str,
                        help='which dataset')
    # parser.add_argument('--src_address', default="/home/dyt/Messi_du/2022/RDA/data/Office-31/amazon_uniform_noisy_0.4.txt", type=str,
    #                     help='address of image list of source dataset')
    # parser.add_argument('--tgt_address', default="/home/dyt/Messi_du/2022/RDA/data/Office-31/dslr.txt", type=str,
    #                     help='address of image list of target dataset')
    parser.add_argument(
        '--stats_file',
        default=
        "/home/dyt/Messi_du/save_code/RDA_copy/statistic/TCL-A2W-uniform-noisy-0.4-2022-02-22-13-37-40.pkl",
        type=str,
        help='store the training loss and and validation acc')
    parser.add_argument('--noisy_rate',
                        default=None,
                        type=float,
                        help='noisy rate')
    #######
    parser.add_argument('--Dataset',
                        default='office31',
                        type=str,
                        help='dataset')
    parser.add_argument('--SourceDataset',
                        default='webcam',
                        type=str,
                        help='source dataset')
    parser.add_argument('--TargetDataset',
                        default='dslr',
                        type=str,
                        help='target dataset')
    parser.add_argument('--noise_level',
                        default=0.4,
                        type=float,
                        help='Noise level')
    parser.add_argument('--noise_type',
                        default='unif',
                        type=str,
                        metavar='M',
                        help='The type of noise.')
    parser.add_argument('--batch_size_source',
                        type=int,
                        default=16,
                        help='Batch Size of source domain.')
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='gpu id.')
    parser.add_argument('--seed',
                        type=int,
                        default=10,
                        help='seed.')
    parser.add_argument('--batch_size_target',
                        type=int,
                        default=16,
                        help='Batch Size of target domain.')
    args = parser.parse_args()

    cfg = Config(args.config)
    print(args)
    # source_file = args.src_address
    # target_file = args.tgt_address
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    

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

    # train_source_clean_loader = load_images(source_file, batch_size=32, is_cen=is_cen, split_noisy=False)
    # train_source_noisy_loader = train_source_clean_loader
    # train_target_loader = load_images(target_file, batch_size=32, is_cen=is_cen)
    # test_target_loader = load_images(target_file, batch_size=32, is_train=False)

    # load data
    source_data, target_data = build_dataset(args, args.Dataset, seed=1)
    source_data_track, target_data_track = build_dataset(args, args.Dataset, seed=1)
    print('loading source dataset...')
    source_loader = torch.utils.data.DataLoader(
        source_data,
        batch_size=args.batch_size_source,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    source_loader_track = torch.utils.data.DataLoader(
        source_data_track,
        batch_size=args.batch_size_source,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    print('loading target dataset...')
    target_loader = torch.utils.data.DataLoader(
        target_data,
        batch_size=args.batch_size_target,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    target_loader_track = torch.utils.data.DataLoader(
        target_data_track,
        batch_size=args.batch_size_target,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'

    optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)

    assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=cfg.init_lr)
    to_dump = train(model_instance,
                    source_loader,
                    target_loader,
                    target_loader,
                    source_loader_track,
                    target_loader_track,
                    group_ratios,
                    max_iter=18000,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    eval_interval=500)
    pickle.dump(to_dump, open(args.stats_file, 'wb'))
