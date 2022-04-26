import numpy as np
import os
from os.path import join as pjoin
import torch
import random
import pickle
from scipy.spatial.distance import cdist

def comput_adjacency_matrix_images(coord, mean_px, pixels):

    coord = coord.reshape(-1, 2)
    feature = np.concatenate((coord, mean_px), axis=1)
    dist = cdist(feature, feature)   
    sigma = 0.1 * np.pi
    A = np.exp(- dist / sigma ** 2) 
    A[np.diag_indices_from(A)] = 0  
    return A

def list_to_torch(data):
    """
    convert to torch
    """
    for i in range(len(data)):
        if data[i] is None:
            continue
        elif isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data


def normalize(x, eps=1e-7):
    return x / (x.sum() + eps)


class FVsp(torch.utils.data.Dataset):

    """
    read data from file, construct the graph data
    """
    def __init__(self,
                 data_dir,
                 split,  #'train', 'val', 'test'
                 # use_mean_px=True,#sp
                 model, # ["FV","FP","FKP"]
                 sp, # 200/150
                 use_coord = False,  
                 use_pixles = False,  
                 use_E_AAD = False,
                 use_t_AAD = False,
                 gt_attn_threshold = 0,
                 attn_coef=None):

   
        self.split = split #'train', 'val', 'test'
        self.is_test = split.lower() in ['test', 'val'] #True/False
        
        with open(pjoin(data_dir, '%s_%ssp_%s_c2.00.pkl' % (model, str(sp), split)), 'rb') as f:
            self.labels, self.sp_data = pickle.load(f)

        self.use_pixles = use_pixles
        self.use_coord = use_coord
        self.use_E_AAD = use_E_AAD
        self.use_t_AAD = use_t_AAD
        self.n_samples = len(self.labels)#graph number

        if model == "FP":
            self.img_size =150#fp 150
        else:
            self.img_size = 200 # FV / FKP 200

        self.gt_attn_threshold = gt_attn_threshold 


        
        self.alpha_WS = None
        if attn_coef is not None and not self.is_test: #not test,val
            with open(attn_coef, 'rb') as f:
                self.alpha_WS = pickle.load(f)
            print('using weakly-supervised labels from %s (%d samples)' % (attn_coef, len(self.alpha_WS)))#从attn_coef加载弱监督标签

    
    def train_val_split(self, samples_idx):
        self.sp_data = [self.sp_data[i] for i in samples_idx]
        self.labels = self.labels[samples_idx]
        self.n_samples = len(self.labels)

    
    def precompute_graph_images(self, num_nodes):
        print('precompute all data for the %s set...' % self.split.upper()) 
        self.Adj_matrices, self.node_features, self.GT_attn, self.WS_attn = [], [], [], [] 
        for index, sample in enumerate(self.sp_data):
            mean_px, coord, pixels = sample[:3] 
            if num_nodes == 0:
                coord = coord / self.img_size
            else:
                coord = coord[:num_nodes] / self.img_size
                mean_px = mean_px[:num_nodes]
                pixels = pixels[:num_nodes]
            A = comput_adjacency_matrix_images(coord, mean_px, pixels) 
            N_nodes = A.shape[0]
            x = None

            x = mean_px.reshape(N_nodes, -1) 
            if self.use_coord:
                coord = coord.reshape(N_nodes, 2) 
                x = np.concatenate((x, coord), axis=1) 
            if self.use_pixles:
                x=np.concatenate((x,pixels),axis=1)

            if x is None:
                x = np.ones((N_nodes, 1))  
            

            if self.gt_attn_threshold == 0:

                gt_attn = (mean_px < 0.75).astype(np.float32)

            else:
               
                gt_attn = mean_px.copy()
                gt_attn[gt_attn > self.gt_attn_threshold] = 0

            self.GT_attn.append(normalize(gt_attn))
            if self.alpha_WS is not None: 
                self.WS_attn.append(normalize(self.alpha_WS[index])) 

            self.node_features.append(x)
            self.Adj_matrices.append(A)
            

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        data = [self.node_features[index],
                self.Adj_matrices[index],
                self.Adj_matrices[index].shape[0],
                self.labels[index],
                self.GT_attn[index]]

        if self.alpha_WS is not None:
            data.append(self.WS_attn[index])

        data = list_to_torch(data)  # convert to torch

        return data


def load_data(args, model = "FV", sp = 150, num_nodes= 0):
    """
    read data from file, construct the graph data
    """
    use_pixles = 'pixles' in args.img_features
    use_coord = 'coord' in args.img_features
    use_E_AAD = 'E_AAD' in args.img_features
    use_t_AAD = 't_AAD' in args.img_features

    if args.train == True:
        train_dataset = FVsp(args.data_dir, split='train', model=model, sp=sp, use_pixles=use_pixles, use_coord=use_coord, use_E_AAD=use_E_AAD, use_t_AAD=use_t_AAD)
        train_dataset.precompute_graph_images(num_nodes)
    else:
        train_dataset = None
    
    test_dataset = FVsp(args.data_dir, split='test', model=model, sp=sp, use_pixles=use_pixles, use_coord=use_coord, use_E_AAD=use_E_AAD, use_t_AAD=use_t_AAD)             
    test_dataset.precompute_graph_images(num_nodes)

    return train_dataset, test_dataset, model


def load_save_noise(f = None, noise_shape = None):
    """load noise or creat noise"""
  
    noises = torch.randn(noise_shape, dtype=torch.float) 
                                                            
        
    return noises / 10.


def Two_data_iter(batch_size, features_fv, adj_fv,
                    features_fp, adj_fp, labels):
    num_examples_fv = len(features_fv)
    num_examples_fp = len(features_fp)

    assert num_examples_fv == num_examples_fp

    indices = list(range(num_examples_fv))

    random.shuffle(indices)
    for i in range(0, num_examples_fv, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples_fv)])
        yield features_fv[batch_indices], adj_fv[batch_indices], features_fp[batch_indices], adj_fp[batch_indices], labels[batch_indices]



def get_length(generator):
    if hasattr(generator, "__len__"):
        return len(generator)
    else:
        return sum(1 for _ in generator)

