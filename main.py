import datetime
from os.path import join
from numpy.core.fromnumeric import size
import torch
import argparse
import pickle
import torch.nn.functional as F
from utils.model import Net, GNN
from utils.load_data import load_data, Two_data_iter
from utils.Fusion import Fusion_block
from utils.metric import *
from tqdm import tqdm


max_nodes = 189


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run experiments with Graph Neural Networks')

    
    parser.add_argument('-D', '--dataset', type=str, default='fvsp')
    parser.add_argument('-s', '--n_sp', type=list,
                        default=[200, 200, 200], help='sp of all models, the order is FV,FKP,FP')
    parser.add_argument('-d', '--data_dir', type=str,
                        default=r"data", help='path to the dataset')
    parser.add_argument('-r', '--results', type=str,
                        default=r"results")
    parser.add_argument('--out_features', type=int,
                        default=585, help='classes')  
    parser.add_argument('-e','--epoch', type=int, default=50, help='n_batches')
    parser.add_argument('--img_features', type=str, default='pixles,coord',
                        help='image features to use as node features')  
    parser.add_argument('-fs', '--fusion_strategy', type=str, default='pagerank')
    parser.add_argument('-B', '--batch_size', type=int, default=54)
    parser.add_argument('--model', type=str, default='CGCN', choices=['GCN', 'CGCN'])
    parser.add_argument('--train', type=bool, default=False, choices=[True, False])
    args = parser.parse_args()

    return args

args = parse_args()

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes

def to_tensor(train_dataset, test_dataset, model, args):
    """array2tensor and load noise""" 
    if args.train == True:
        train_x = train_dataset.node_features
        train_adj = train_dataset.Adj_matrices
        train_y = train_dataset.labels

        train_x = torch.tensor(train_x, dtype=torch.float32)
        if model == 'FP':
            train_x = torch.where(torch.isnan(train_x),
                                torch.full_like(train_x, 0), train_x)
            train_x = torch.tensor(train_x, dtype=torch.float32)

        train_adj = torch.tensor(train_adj, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.int64)
    else:
        train_x = None
        train_adj = None
        train_y = None

    test_x = test_dataset.node_features
    test_adj = test_dataset.Adj_matrices
    test_y = test_dataset.labels

    test_x = torch.tensor(test_x, dtype=torch.float32)
    if model == 'FP':
        test_x = torch.where(torch.isnan(
            test_x), torch.full_like(test_x, 0), test_x)
        test_x = torch.tensor(test_x, dtype=torch.float32)

    test_adj = torch.tensor(test_adj, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.int64)

    return train_x, train_adj, train_y, test_x, test_adj, test_y


############################load data#########################
# get fv graph features

train_fv_dataset, test_fv_dataset, model = load_data(
    args, model="FV", sp=args.n_sp[0])

train_x_fv, train_adj_fv, train_y_fv, test_x_fv, test_adj_fv, test_y_fv = to_tensor(
    train_fv_dataset, test_fv_dataset, model, args)

# get fp graph features
train_fp_dataset, test_fp_dataset, model = load_data(
    args, model="FP", sp=args.n_sp[2])

train_x_fp, train_adj_fp, train_y_fp, test_x_fp, test_adj_fp, test_y_fp = to_tensor(
    train_fp_dataset, test_fp_dataset, model, args)

if args.train == True:
    print(train_x_fv.size(), train_x_fp.size())
print(test_x_fv.size(), test_x_fp.size())

n = len(test_fv_dataset)
print(f'current fusion mode:{args.fusion_strategy}')
if args.train:
    print(f'current statement: Train.')
else:
    print(f'current statement: Test.')

# device = "cpu"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################################################

class DataParallel(torch.nn.Module):
    def __init__(self, batch_size, num_nodes, CONV = 'CGCN'):
        super(DataParallel, self).__init__()

       
        self.net_FV = Net(CONV)

        self.gnn_afterFusion = GNN(100 * 6, 400, 400, add_loop=True, lin=False)
        self.net_FP = Net(CONV)

        self.lin1_afterFusion = torch.nn.Linear(2 * 400 + 400, 700)
        self.lin2_afterFusion = torch.nn.Linear(700, 585)

        self.fusion = Fusion_block(args.fusion_strategy, batch_size, num_nodes, device)

    def forward(self, x_FV, adj_FV, x_FP, adj_FP, mask=None):

        x1, adj1, _, _ = self.net_FV(x_FV, adj_FV)

        x3, adj3, _, _ = self.net_FP(x_FP, adj_FP)

        x1, adj1, x3, adj3 = self.fusion(x1, adj1, x3, adj3)
        x = torch.cat([x1, x3], dim=-1)
        adj = adj1 + adj3
        ##############

        ####afterFusion####
        x = self.gnn_afterFusion(x, adj, mask)
        x = x.mean(dim=1)
        x = F.relu(self.lin1_afterFusion(x))
        x = self.lin2_afterFusion(x)
        ###################

        return F.log_softmax(x, dim=-1)

net = DataParallel(args.batch_size, 28, args.model).to(device)


if not args.train:
    net.load_state_dict(torch.load("model/model_dict.pth"))
    net.eval()

optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)


def train(epoch):
    net.train()
    loss_all = 0

    for x_v, adj_v, x_p, adj_p, y in Two_data_iter(args.batch_size, train_x_fv, train_adj_fv, train_x_fp, train_adj_fp, train_y_fv):
        x_v = x_v.to(device)
        adj_v = adj_v.to(device)

        x_p = x_p.to(device)
        adj_p = adj_p.to(device)

        y = y.to(device)
        optimizer.zero_grad()
        output = net(x_v, adj_v, x_p, adj_p)
        loss = F.nll_loss(output, y)
        loss.backward()
        loss_all += y.size(0) * loss.item()
        optimizer.step()

    return loss_all / len(train_x_fv)


@torch.no_grad()
def test(x_fv, adj_fv, x_fp, adj_fp, y_fv):
    """
    return: acc, pre(all result of net), targets(real laberls)
    """
    net.eval()
    correct = 0
    length = 0
    pre, targets = [], []

    for x_v, adj_v, x_p, adj_p, y in tqdm(Two_data_iter(args.batch_size, x_fv, adj_fv, x_fp, adj_fp, y_fv)):
        x_v = x_v.to(device)
        adj_v = adj_v.to(device)
        x_p = x_p.to(device)
        adj_p = adj_p.to(device)
        y = y.to(device)
        
        out = net(x_v, adj_v, x_p, adj_p)

        pre.append(out.detach())
        targets.append(y.detach())

        prediction = out.max(dim=1)[1]
        correct += prediction.eq(y).sum().item()

    pre = torch.cat(pre)
    targets = torch.cat(targets)

    return correct / n, pre, targets


best_val_acc = test_acc = 0
best_test_acc = 80
train_loss_all = []
test_acc_all = []
ROC = []
if args.train:
    for epoch in range(1, args.epoch):
        
        train_loss = train(epoch)
        test_acc, pre, targets = test(test_x_fv, test_adj_fv, test_x_fp,
                                        test_adj_fp, test_y_fv)
        if args.train == True and test_acc*100 > best_test_acc:
            torch.save(net.state_dict(), "model/model_dict_{:.2f}.pth".format(test_acc * 100))
            best_test_acc = test_acc*100

        if epoch == args.epoch-1:
            th, fpr, tpr, roc_auc = acc_roc(pre, targets)
            ROC.append(th)
            ROC.append(fpr)
            ROC.append(tpr)
            ROC.append(roc_auc)

            n_sp = ''.join((list(map(str, args.n_sp))))  # [1,2,3] to '123'
            with open(r'%s/%ssp_%s_ROC.pkl' % (args.results, n_sp, args.fusion_strategy), 'wb') as f:
                pickle.dump(ROC, f, protocol=2)

        train_loss_all.append(train_loss)
        test_acc_all.append(test_acc)

        print('Epoch: {:03d}, Train Loss: {:.7f}, '
            ' Test Acc: {:.2f}%'.format(epoch, train_loss,
                                        test_acc*100))
else:
    test_acc, pre, targets = test(test_x_fv, test_adj_fv, test_x_fp,
                                        test_adj_fp, test_y_fv)
    print('Test Acc: {:.2f}%'.format(test_acc*100))