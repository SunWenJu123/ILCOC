import numpy as np

import torch
from torch.optim import SGD
from utils.buffer import Buffer
from torch.utils.data import DataLoader

from models.utils.continual_model import ContinualModel

import matplotlib.pyplot as plt
import seaborn as sns

def get_backbone(bone, embedding_dim, output_size=None, nf=64):
    from backbone.MNISTMLP import MNISTMLP
    from backbone.MNISTMLP_OC import MNISTMLP_OC
    from backbone.ResNet18 import ResNet
    from backbone.ResNet18_OC import resnet18_OC

    if output_size is None:
        output_size = bone.output_size

    if isinstance(bone, MNISTMLP):
        return MNISTMLP_OC(bone.input_size, embedding_dim, output_size)
    elif isinstance(bone, ResNet):
        return resnet18_OC(embedding_dim, output_size, nf)
    else:
        raise NotImplementedError('Progressive Neural Networks is not implemented for this backbone')


class OCILFAST(ContinualModel):
    NAME = 'OCILFAST'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, net, loss, args, transform):
        super(OCILFAST, self).__init__(net, loss, args, transform)

        self.nets = []
        self.c = []
        self.threshold = []

        self.nu = self.args.nu
        self.eta = self.args.eta
        self.eps = self.args.eps
        self.embedding_dim = self.args.embedding_dim
        self.weight_decay = self.args.weight_decay
        self.margin = self.args.margin

        self.current_task = 0
        self.cpt = None
        self.nc = None
        self.eye = None
        self.buffer_size = self.args.buffer_size
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.nf = self.args.nf

        if self.args.dataset == 'seq-cifar10' or self.args.dataset == 'seq-mnist':
            self.input_offset = -0.5
        elif self.args.dataset == 'seq-tinyimg':
            self.input_offset = 0
        else:
            self.input_offset = 0

    # 任务初始化
    def begin_task(self, dataset):

        if self.cpt is None:
            self.cpt = dataset.N_CLASSES_PER_TASK
            self.nc = dataset.N_TASKS * self.cpt
            self.eye = torch.tril(torch.ones((self.nc, self.nc))).bool().to(self.device)  # 下三角包括对角线为True，上三角为False，用于掩码

        if len(self.nets) == 0:
            for i in range(self.nc):
                self.nets.append(
                    get_backbone(self.net, self.embedding_dim, self.nc, self.nf).to(self.device)
                )
                self.c.append(
                    torch.ones(self.embedding_dim, device=self.device)
                )

        self.current_task += 1

    def train_model(self, dataset, train_loader):

        categories = list(range((self.current_task - 1) * self.cpt, (self.current_task) * self.cpt))
        print('==========\t task: %d\t categories:' % self.current_task, categories, '\t==========')
        if self.args.print_file:
            print('==========\t task: %d\t categories:' % self.current_task, categories, '\t==========', file=self.args.print_file)

        for category in categories:
            losses = []

            if category > 0:
                self.reset_train_loader(train_loader, category)

            for epoch in range(self.args.n_epochs):

                avg_loss, maxloss, posdist, negdist, gloloss = self.train_category(train_loader, category, epoch)

                losses.append(avg_loss)
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    print("epoch: %d\t task: %d \t category: %d \t loss: %f" % (
                        epoch + 1, self.current_task, category, avg_loss))

                    if self.args.print_file:
                        print("epoch: %d\t task: %d \t category: %d \t loss: %f" % (
                            epoch + 1, self.current_task, category, avg_loss), file=self.args.print_file)

                    plt.figure(figsize=(20, 12))

                    ax = plt.subplot(2, 2, 1)
                    ax.set_title('maxloss')
                    plt.xlim((0, 2))
                    if maxloss is not None:
                        try:
                            sns.distplot(maxloss)
                        except:
                            pass

                    ax = plt.subplot(2, 2, 2)
                    ax.set_title('posdist')
                    plt.xlim((0, 2))
                    try:
                        sns.distplot(posdist)
                    except:
                        print(posdist)

                    ax = plt.subplot(2, 2, 3)
                    ax.set_title('negdist')
                    plt.xlim((0, 2))
                    try:
                        sns.distplot(negdist)
                    except:
                        print(negdist)

                    ax = plt.subplot(2, 2, 4)
                    ax.set_title('gloloss')
                    plt.xlim((0, 2))
                    try:
                        sns.distplot(gloloss)
                    except:
                        print(gloloss)

                    plt.savefig("../"+self.args.img_dir+"/loss-cat%d-epoch%d.png" % (category, epoch))
                    plt.clf()

            x = list(range(len(losses)))
            plt.plot(x, losses)
            plt.savefig("../"+self.args.img_dir+"/loss-cat%d.png" % (category))
            plt.clf()

        self.fill_buffer(train_loader)


    def reset_train_loader(self, train_loader, category):

        dataset = train_loader.dataset
        input = dataset.data
        loader = DataLoader(dataset,
                              batch_size=self.args.batch_size, shuffle=False)

        inputs = []
        targets = []
        prev_dists = []

        prev_categories = list(range(category))
        print('prev_categories', prev_categories)
        if self.args.print_file:
            print('prev_categories', prev_categories, file=self.args.print_file)
        for i, data in enumerate(loader):
            input, target, _ = data
            _, prev_dist = self.predict(input, prev_categories)

            inputs.append(input.detach().cpu())
            targets.append(target.detach().cpu())
            prev_dists.append(prev_dist.detach().cpu())

        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)
        prev_dists = torch.cat(prev_dists, dim=0)
        dataset.set_prevdist(prev_dists)

    def train_category(self, data_loader, category: int, epoch_id):

        self.init_center_c(data_loader, category)
        c = self.c[category]

        network = self.nets[category].to(self.device)
        network.train()

        optimizer = SGD(network.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        avg_loss = 0.0
        sample_num = 0

        maxloss = []
        posdist = []
        negdist = []
        gloloss = []

        prev_categories = list(range(category))
        for i, data in enumerate(data_loader):
            inputs, semi_targets, prev_dists = data
            inputs = inputs.to(self.device)
            semi_targets = semi_targets.to(self.device)
            prev_dists = prev_dists.to(self.device)

            if (not self.buffer.is_empty()) and self.args.buffer_size > 0:
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform)
                # print(buf_inputs[0])
                inputs = torch.cat((inputs, buf_inputs))
                semi_targets = torch.cat((semi_targets, buf_labels))

            # Zero the network parameter gradients
            optimizer.zero_grad()

            # 注意网络的输入要减去0.5
            outputs = network(inputs + self.input_offset)

            dists = torch.sum((outputs - c) ** 2, dim=1)
            pos_dist_loss = torch.relu(dists - self.args.r)

            if category > 0 :
                max_scores = torch.relu(dists.view(-1, 1) - prev_dists)
                max_loss = torch.sum(max_scores, dim=1) * self.margin / category

                loss_pos = pos_dist_loss + max_loss
                loss_neg = self.eta * dists ** -1

                pos_max_loss = max_loss[semi_targets == category]
                maxloss.append(pos_max_loss.detach().cpu().data.numpy())

            else:
                loss_pos = pos_dist_loss
                loss_neg = self.eta * dists ** -1

            losses = torch.where(semi_targets == category, loss_pos, loss_neg)
            gloloss.append(losses.detach().cpu().data.numpy())
            loss = torch.mean(losses)

            loss.backward()
            optimizer.step()

            # 记录损失部分
            pos_dist = pos_dist_loss[semi_targets == category]
            posdist.append(pos_dist.detach().cpu().data.numpy())

            neg_dist = loss_neg[semi_targets != category]
            negdist.append(neg_dist.detach().cpu().data.numpy())

            avg_loss += loss.item()
            sample_num += inputs.shape[0]

            # 旧类别只训练一次
            if category < (self.current_task - 1) * self.cpt:
                break

        avg_loss /= sample_num
        if len(maxloss) > 0:
            maxloss = np.hstack(maxloss)
        else: maxloss = None
        posdist = np.hstack(posdist)
        negdist = np.hstack(negdist)
        gloloss = np.hstack(gloloss)
        return avg_loss, maxloss, posdist, negdist, gloloss


    def fill_buffer(self, train_loader):
        for data in train_loader:
            # get the inputs of the batch
            inputs, semi_targets, not_aug_inputs = data
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=semi_targets)

    def init_center_c(self, train_loader: DataLoader, category):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = 0

        net = self.nets[category].to(self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, semi_targets, not_aug_inputs = data
                inputs = inputs.to(self.device)
                semi_targets = semi_targets.to(self.device)
                outputs = net(inputs + self.input_offset)
                outputs = outputs[semi_targets == category]  # 取所有正样本来进行圆心初始化
                # print(outputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < self.eps) & (c < 0)] = -self.eps
        c[(abs(c) < self.eps) & (c > 0)] = self.eps
        self.c[category] = c.to(self.device)


    def get_score(self, dist, category):
        score = 1 / (dist + 1e-6)

        return score

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        categories = list(range(self.current_task * self.cpt))
        return self.predict(x, categories)[0]

    def predict(self, inputs: torch.Tensor, categories):
        inputs = inputs.to(self.device)
        outcome, dists = [], []
        with torch.no_grad():
            for i in categories:
                net = self.nets[i]
                net.to(self.device)
                net.eval()

                c = self.c[i].to(self.device)

                pred = net(inputs + self.input_offset)
                dist = torch.sum((pred - c) ** 2, dim=1)

                scores = self.get_score(dist, i)

                outcome.append(scores.view(-1, 1))
                dists.append(dist.view(-1, 1))

        outcome = torch.cat(outcome, dim=1)
        dists = torch.cat(dists, dim=1)
        return outcome, dists
