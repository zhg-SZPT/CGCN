# Compute superpixels for yeiziFV using SLIC algorithm
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic

import numpy as np
import random
import os
import scipy
import pickle
from os.path import join as pjoin
from skimage.segmentation import slic
from skimage import filters
from torchvision import datasets
import multiprocessing as mp
import scipy.ndimage
import scipy.spatial
import argparse
import datetime
from utils.Data import *
from skimage import feature
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

import operator
from functools import reduce



def parse_args():
    parser = argparse.ArgumentParser(description='Extract SLIC superpixels from images')
    parser.add_argument('-D', '--dataset', type=str, default='FP',choices=['FV', 'FP','FKP'])
    parser.add_argument('-d1', '--data_dir1', type=str, default=r'/home/qu/paper_code/diff_PYG/data/fp100/585class/o',help='path to the ori_dataset')
    parser.add_argument('-d2', '--data_dir2', type=str, default=r'/home/qu/paper_code/diff_PYG/data/fp100/585class/e',help='path to the enhance_dataset')
    parser.add_argument('-d3', '--data_dir3', type=str, default=r'/home/qu/paper_code/diff_PYG/data/fp100/585class/t',help='path to the thin_dataset')
    parser.add_argument('-o', '--out_dir', type=str, default=r'/home/qu/paper_code/diff_PYG/data/data_file585', help='path where to save superpixels')
    parser.add_argument('-s', '--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('-t', '--threads', type=int, default=4 , help='number of parallel threads')
    parser.add_argument('-n', '--n_sp', type=int, default=500, help='max number of superpixels per image')#(91,200) #1150
    parser.add_argument('-c', '--compactness', type=float, default=2, help='compactness of the SLIC algorithm '
                                                                            '(Balances color proximity and space proximity')#0.6
    parser.add_argument('--seed', type=int, default=111, help='seed for shuffling nodes')
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args

def normalize_zero_one(im):
    m1 = im.min()
    m2 = im.max()
    return (im - m1) / (m2 - m1)

def process_image(params):
    img_o,img_e,img_t, index, n_images, args, to_print, shuffle = params 

    assert img_e.dtype == np.uint8, img.dtype
    assert img_o.dtype == np.uint8, img.dtype
    img_o = (img_o / 255.)  
    img_e_n = (img_e / 255.)
    img_t = (img_t/255.).astype(np.bool)


    n_sp_extracted = args.n_sp + 1  
    n_sp_query = args.n_sp + 20
    while n_sp_extracted > args.n_sp:
        superpixels = slic(img_e_n, n_segments=n_sp_query, compactness=args.compactness, multichannel=False)
        sp_indices = np.unique(superpixels)
        n_sp_extracted = len(sp_indices)
        n_sp_query -= 1  

    assert n_sp_extracted <= args.n_sp and n_sp_extracted > 0, (args.split, index, n_sp_extracted, args.n_sp)
    assert n_sp_extracted == np.max(superpixels) + 1, ('superpixel indices', np.unique(superpixels))  
    n_sp = n_sp_extracted

    if shuffle:
        ind = np.random.permutation(n_sp_extracted) 
    else:
        ind = np.arange(n_sp_extracted) 

    sp_order = sp_indices[ind].astype(np.int32)

    if len(img_o.shape) == 2:
        img_o = img_o[:, :, None]

    n_ch = 1 if img_o.shape[2] == 1 else 3 
    
    sp_piexls = []
    for i in np.unique(superpixels):
        sp_piexls.append(np.sum(superpixels == i))

    sp_intensity, sp_coord= [], []

    
    for seg in sp_order:
        
        mask = (superpixels == seg).squeeze()
        avg_value = np.zeros(n_ch)
        for c in range(n_ch):
            avg_value[c] = np.mean(img_o[:, :, c][mask]) 
        cntr = np.array(scipy.ndimage.measurements.center_of_mass(mask))  
        sp_intensity.append(avg_value)
        sp_coord.append(cntr)


    sp_intensity = np.array(sp_intensity, np.float32)
    sp_coord = np.array(sp_coord, np.float32)
    sp_piexls= np.array(sp_piexls, np.float32)
    sp_piexls = np.resize(sp_piexls, (n_sp, 1))
    sp_piexls = normalize_zero_one(sp_piexls)

    if to_print:
        print('image={}/{}, shape={}, min={:.2f}, max={:.2f}, n_sp={}'.format(index +1, n_images, img_e_n.shape,
                                                                              img_e_n.min(), img_e_n.max(),
                                                                              sp_intensity.shape[0]))

    return sp_intensity, sp_coord, sp_piexls, sp_order, superpixels, n_sp_extracted


if __name__ == '__main__':

    dt = datetime.datetime.now()
    print('start time:', dt)

    args = parse_args()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed) 

    is_train = args.split.lower() == 'train'
    if args.dataset == 'FV' or args.dataset =='FKP':
        Fdata_o = FV(args.data_dir1, train=is_train)
        Fdata_e = FV(args.data_dir2, train=is_train)
        Fdata_t = FV(args.data_dir3, train=is_train)


        assert args.n_sp > 1 and args.n_sp < 200 * 91, (
            'the number of superpixels cannot exceed the total number of pixels or be too small')  #200*91
    else:
        
        Fdata_o = FP(args.data_dir1, train=is_train)
        Fdata_e = FP(args.data_dir2, train=is_train)
        Fdata_t = FP(args.data_dir3, train=is_train)


    images_o = Fdata_o.data
    images_e = Fdata_e.data
    images_t = Fdata_t.data
    labels = Fdata_o.label
    if not isinstance(images_o, np.ndarray): 
        images_o = images_o.numpy()
    if not isinstance(images_e, np.ndarray):
        images_e = images_e.numpy()
    if not isinstance(images_t, np.ndarray):
        images_t = images_t.numpy()
    if isinstance(labels, list):
        labels = np.array(labels)
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy()

    n_images = len(labels)
    n=[]

    if args.threads <= 0:
        sp_data = []
        for i in range(n_images):
            sp_data.append(process_image((images_o[i],images_e[i],images_t[i], i, n_images, args, True, False))) #process_image() return sp_intensity, sp_coord, sp_order, superpixels #超像素的强度、坐标、序号、像素值
    else:
        with mp.Pool(processes=args.threads) as pool:
            sp_data = pool.map(process_image, [(images_o[i],images_e[i],images_t[i], i, n_images, args, True, False) for i in range(n_images)])


    superpixels = [sp_data[i][4] for i in range(n_images)] 
    n=[sp_data[i][5] for i in range(n_images)]
    print(max(n))
    sp_data = [sp_data[i][:4] for i in range(n_images)]

    with open('%s/%s_%dsp_%s_c%.2f.pkl' % (args.out_dir, args.dataset, args.n_sp, args.split,args.compactness), 'wb') as f:
        pickle.dump((labels.astype(np.int32), sp_data), f, protocol=2) #labels，sp_data
    with open('%s/%s_%dsp_%s_c%.2f_superpixels.pkl' % (args.out_dir, args.dataset, args.n_sp, args.split,args.compactness), 'wb') as f:
        pickle.dump(superpixels, f, protocol=2)#superpixels

    print('done in {}'.format(datetime.datetime.now() - dt))



