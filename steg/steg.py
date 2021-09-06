import os
import random

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from torch.autograd import Variable
from torch.utils.data import DataLoader

import steg.transformed as transforms
from data import MyImageFolder, NewMyImageFolder
from models import UnetGenerator
from models import RevealNet
from utils import msssim

from utils.AFD_config import opt


location_map = {
    'transfer':opt.cover_dir,
    'ImageNet':'/mnt/disk50/datasets/ImageNet/val',
    'LFW':'/mnt/data2/pengyi/datasets/lfw',
    'VOC2012':'/mnt/disk50/datasets/VOC2012/JPEGImages/test'
    }


def init(cover_type='transfer'):
    assert cover_type in ['transfer', 'ImageNet', 'LFW', 'VOC2012'], 'Expect dataset in [transfer, ImageNet, LFW, VOC2012], but got ' + cover_type
    cudnn.benchmark = True

    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Hnet.apply(weights_init)
    opt.imageSize = 256
    
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()

    if cover_type in ['transfer', 'VOC2012']:
        cover_dataset = MyImageFolder(
            location_map[cover_type],
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),
                transforms.ToTensor(),
            ]))
    else:
        cover_dataset = NewMyImageFolder(
            location_map[cover_type],
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),
                transforms.ToTensor(),
            ]))

    return Hnet, cover_dataset

def steg_forward(Hnet, cover_dataset, secret_img, img_size, device, enable, ftype='normal', cover_idx=False):
    assert ftype in ['normal', 'special']

    secret_img = secret_img.cpu()
    batch_size = secret_img.size()[0]
    batch_idx = []
    cover_img = []
    idxs = np.random.randint(0, len(cover_dataset) -1, batch_size)
    for i in range(batch_size):
        if i == 0:
            cover_img.append(torch.unsqueeze(cover_dataset[idxs[i]], 0))
        else:
            cover_tensor = torch.unsqueeze(cover_dataset[idxs[i]], 0)
            cover_img.append(cover_tensor)
        if cover_idx:
            batch_idx.append(idxs[i])
    cover_img = torch.cat(cover_img, 0)

    concat_img = torch.cat([cover_img, secret_img], dim=1)
    concat_img = concat_img

    with torch.no_grad():
        secret_imgv = Variable(secret_img)
        cover_imgv = Variable(cover_img).cuda()
        concat_imgv = Variable(concat_img).cuda()

    container_img = Hnet(concat_imgv)

    lossH = 1 - msssim(container_img, cover_imgv, normalize=True)

    if cover_idx:
        if ftype == 'special':
            return container_img, lossH, cover_img, batch_idx
        return container_img, lossH, batch_idx
    else:
        if ftype == 'special':
            return container_img, lossH, cover_img.cuda()
        return container_img, lossH


def steg_forward_vgg(Hnet, cover_dataset, secret_img, opt, enable):

    secret_img = secret_img.cpu()
    batch_size = secret_img.size()[0]

    for i in range(batch_size):
        if i == 0:
            idx = random.randint(0, len(cover_dataset) - 1)
            cover_img = torch.unsqueeze(cover_dataset[idx], 0)
        else:
            idx = random.randint(0, len(cover_dataset) - 1)
            cover_tensor = torch.unsqueeze(cover_dataset[idx], 0)
            cover_img = torch.cat([cover_img, cover_tensor], 0)

    secret_img = tensor_image_resize(secret_img, 256).cuda()
    cover_img = tensor_image_resize(cover_img, 256).cuda()

    secret_img.requires_grad = True
    cover_img.requires_grad = True

    concat_img = torch.cat([cover_img, secret_img], dim=1)
    concat_img = concat_img.cuda()

    with torch.no_grad():
        # secret_imgv = Variable(secret_img).cuda()
        cover_imgv = Variable(cover_img).cuda()
        concat_imgv = Variable(concat_img).cuda()

    container_img = Hnet(concat_imgv)
    cover_imgv.requires_grad = True
    lossH = 1 - msssim(container_img.cuda(), cover_imgv.cuda(), normalize=True)
    return container_img, lossH


def steg_forward_vggm(Hnet, cover_dataset, secret_img, label, cover_seed, opt, enable):

    secret_img = secret_img.cpu()
    batch_size = secret_img.size()[0]
    cover_index = []

    for i in range(batch_size):
        if i == 0:
            idx = random.randint(0, len(cover_dataset) - 1)
            cover_img = torch.unsqueeze(cover_dataset[idx], 0)
        else:
            idx = random.randint(0, len(cover_dataset) - 1)
            cover_tensor = torch.unsqueeze(cover_dataset[idx], 0)
            cover_img = torch.cat([cover_img, cover_tensor], 0)
        cover_index.append(idx)

    cover_img = cover_img.cuda()
    secret_img = secret_img.cuda()
    secret_img.requires_grad = True
    cover_img.requires_grad = True

    concat_img = torch.cat([cover_img, secret_img], dim=1)
    concat_img = concat_img.cuda()

    with torch.no_grad():
        cover_imgv = Variable(cover_img).cuda()
        concat_imgv = Variable(concat_img).cuda()

    concat_imgv.requires_grad = True

    container_img = Hnet(concat_imgv)
    cover_imgv.requires_grad = True
    lossH = 1 - msssim(container_img.cuda(), cover_imgv.cuda(), normalize=True)
    return container_img, lossH, cover_index, cover_img


def reveal_forward(Rnet, secret_img, container_embed, opt, enable):

    with torch.no_grad():
        secret_imgv = Variable(secret_img).cuda()

    reveal_img = Rnet(container_embed)

    criterion = nn.MSELoss().cuda()
    secret_imgv.requires_grad = True
    lossR = criterion(reveal_img.cuda(), secret_imgv.cuda())
    return reveal_img, lossR


def reveal_forward_vgg(Rnet, secret_img, container_embed, opt, enable):
    secret_imgv = secret_img.cuda()
    reveal_img = Rnet(container_embed)
    secret_imgv.requires_grad = True
    lossR = 1 - msssim(reveal_img.cuda(), secret_imgv.cuda(), normalize=True)
    return reveal_img, lossR


def steg_validation(Hnet, Rnet, cover_dataset, secret_img, img_size, device):
    Hnet.train()
    Rnet.train()
    batch_size = secret_img.size()[0]

    for i in range(batch_size):
        if i == 0:
            cover_img = torch.unsqueeze(cover_dataset[random.randint(0, len(cover_dataset) - 1)], 0)
        else:
            cover_tensor = torch.unsqueeze(cover_dataset[random.randint(0, len(cover_dataset) - 1)], 0)
            cover_img = torch.cat([cover_img, cover_tensor], 0)

    secret_img = tensor_image_resize(secret_img).cuda()
    cover_img = tensor_image_resize(cover_img).cuda()

    concat_img = torch.cat([cover_img, secret_img], dim=1)
    concat_img = concat_img.cuda()

    with torch.no_grad():
        secret_imgv = Variable(secret_img)
        concat_imgv = Variable(concat_img)

    container_img = Hnet(concat_imgv)
    reveal_img = Rnet.forward_reveal(container_img)

    lossH = criterion(container_img, cover_imgv)
    lossR = criterion(reveal_img, secret_imgv)

    return val_sumloss, val_rloss


def steg_op(Hnet, Rnet, cover_dataset, secret_img, img_size, device):
    # def steg_op(secret_img, img_size):
    secret_img = secret_img.cpu()
    batch_size = secret_img.size()[0]

    secret_img = secret_normalize(secret_img)

    for i in range(batch_size):
        if i == 0:
            cover_img = torch.unsqueeze(cover_dataset[random.randint(0, len(cover_dataset) - 1)], 0)
        else:
            cover_tensor = torch.unsqueeze(cover_dataset[random.randint(0, len(cover_dataset) - 1)], 0)
            cover_img = torch.cat([cover_img, cover_tensor], 0)

    secret_img = tensor_image_resize(secret_img)
    cover_img = tensor_image_resize(cover_img)

    concat_img = torch.cat([cover_img, secret_img], dim=1)
    concat_img = concat_img.cuda()

    with torch.no_grad():
        concat_imgv = Variable(concat_img)

    print('concat_img: ' + str(concat_imgv.shape))

    container_img = Hnet(concat_imgv)
    reveal_img = Rnet.forward_reveal(container_img)
    container_img = torch.squeeze(container_img, 0).cpu()
    reveal_img = torch.squeeze(reveal_img, 0).cpu()

    container_img = tensor_image_resize(container_img)
    reveal_img = tensor_image_resize(reveal_img)

    save_image(container_img, './container_img.png')
    save_image(reveal_img, './reveal_img.png')
    print("image saved: ./container_img.png")
    print("image saved: ./reveal_img.png")

    return container_img, reveal_img


def tensor_image_resize(img):
    to_PIL = transforms.ToPILImage()
    transform_resize = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ]
    )
    for i in range(img.size()[0]):
        if i == 0:
            res_img = to_PIL(img[i])
            res_img = transform_resize(res_img)
            res_img = torch.unsqueeze(res_img, 0)
        else:
            res_tensor = transform_resize(to_PIL(img[i]))
            res_tensor = torch.unsqueeze(res_tensor, 0)
            res_img = torch.cat([res_img, res_tensor], 0)
    return res_img


def secret_normalize(img):
    secret_norm = transforms.Compose([
        transforms.Normalize([-1, -1, -1], [2, 2, 2])
    ]
    )
    for i in range(img.size()[0]):
        img[i] = secret_norm(img[i])
    return img


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(str(net))


def save_image(resultImg, resultImgName):
    vutils.save_image(resultImg, resultImgName, nrow=1, padding=1, normalize=True)