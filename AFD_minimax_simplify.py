import os
import sys
import copy
import torch
import numpy as np
import time
import math
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn.modules.distance import PairwiseDistance
from torch.autograd import Variable
import torchvision.utils as vutils

sys.path.append(os.path.abspath('./steg'))
import steg
import models
from models import UnetGenerator, FaceNetInceptionModel
from data import MyImageFolder
from data import get_dataloader
from utils import TripletLoss, AverageMeter, Logger
from utils import msssim, evaluate, preprocess_ckpt
from utils.AFD_config import opt
from noise_argparser import NoiseArgParser
from noise_layers.noiser import Noiser
from torch.nn.parallel.data_parallel import DataParallel


# @profile
def train(status='new', **kwargs):
    assert status in ['continue', 'new']

    opt.load_steg_model_dir = './model_zoo/steg_simplify_joint_model.pth'
    opt.load_classifier_model_dir1 = './model_zoo/steg_simplify_joint_model.pth'
    opt.load_classifier_model_dir2 = './model_zoo/steg_simplify_joint_model.pth'
    opt.load_classifier_model_dir = './model_zoo/steg_base_model.pth'
        
    opt.root_dir = './results/'
    opt.logs_dir = opt.root_dir + 'logs'
    opt.classifier_ckpt = opt.root_dir + 'ckpt'

    if not os.path.exists('visualize'):
        os.mkdir('visualize')
    if not os.path.exists(opt.root_dir):
        os.mkdir(opt.root_dir)
    if not os.path.exists(opt.classifier_ckpt):
        os.mkdir(opt.classifier_ckpt)

    opt.parse(kwargs)
    opt.device = 'cuda:0'
    print(opt.device)
    
    NoiseArgParser()(opt.noise, opt)
    
    modelA = FaceNetInceptionModel(opt.embedding_size, opt.num_classes)
    modelA = modelA.cuda()
    modelB = FaceNetInceptionModel(opt.embedding_size, opt.num_classes)
    modelB = modelB.cuda()
    noiser = Noiser(opt.noise, opt.device)
    noiser = noiser.cuda()
    model = FaceNetInceptionModel(opt.embedding_size, opt.num_classes, pretrained=False)
    model = model.cuda()
    triplet = TripletLoss(opt.margin).cuda()


    model.eval()
    
    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Hnet = Hnet.cuda()
    
    cover_dataset = MyImageFolder(
        opt.cover_dir,
        transforms.Compose([
            transforms.Resize([opt.image_size, opt.image_size]),
            transforms.ToTensor(),
        ]))
    
    optimizerH = optim.SGD(Hnet.parameters(), lr=opt.learning_rate * 0.5, momentum=0.9, weight_decay=5e-4)
    optimizerD1 = optim.SGD(modelA.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizerD2 = optim.SGD(modelB.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=5e-4)

    if opt.load_steg_model_dir != '':
        checkpoint = torch.load(opt.load_steg_model_dir, map_location={'cuda:1':opt.device})
        Hnet.load_state_dict(checkpoint['Hnet_state_dict'])

    if opt.load_classifier_model_dir != '':
        checkpoint = torch.load(opt.load_classifier_model_dir, map_location={'cuda:3':opt.device})
        model.load_state_dict(checkpoint['state_dict'])

    if opt.load_classifier_model_dir1 != '':
        checkpoint = torch.load(opt.load_classifier_model_dir1, map_location={'cuda:1':opt.device})
        modelA.load_state_dict(checkpoint['state_dict'])

    if opt.load_classifier_model_dir2 != '':
        checkpoint = torch.load(opt.load_classifier_model_dir2, map_location={'cuda:1':opt.device})
        modelB.load_state_dict(checkpoint['state_dict'])

    modelA = DataParallel(modelA)
    modelB = DataParallel(modelB)
    noiser = DataParallel(noiser)
    model = DataParallel(model)
    Hnet = DataParallel(Hnet)

    if status == 'new':
        logger = Logger(opt.logs_dir, True)
    else:
        logger = Logger(opt.logs_dir)

    mse_loss = torch.nn.MSELoss()
    l2_dist = PairwiseDistance(2)

    cur_lossH = AverageMeter()
    save_lossH = AverageMeter()

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):
        modelA.train()
        modelB.train()
        Hnet.train()
        noiser.train()
        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch, opt.num_epochs + opt.start_epoch - 1))
        # print("learingingH rate: {}".format(optimizerH.param_groups[0]['lr']))
        # print("learingingD1 rate: {}".format(optimizerD1.param_groups[0]['lr']))
        # print("learingingD2 rate: {}".format(optimizerD2.param_groups[0]['lr']))
        data_loaders, data_size = get_dataloader(opt.train_root_dir, opt.valid_root_dir,
                                                 opt.train_csv_name, opt.valid_csv_name,
                                                 opt.num_train_triplets, opt.num_valid_triplets,
                                                 opt.batch_size, opt.num_workers)

        if epoch % 2 == 1:
            torch.cuda.synchronize()
            tic = time.time()
            labelsA, distancesA = [], []
            labelsB, distancesB = [], []
            I_cross_entropy_loss_sumA = 0.0
            I_cross_entropy_loss_sumB = 0.0
            I_triplet_loss_sumA = 0.0
            I_triplet_loss_sumB = 0.0
            I_loss_sumA = 0.0
            I_loss_sumB = 0.0
            I_lossH_sum = 0.0
            I_lossADV_sum = 0.0
            I_content_loss_sum = 0.0
            I_total_loss_sum = 0.0

            for batch_idx, batch_sample in enumerate(data_loaders['train']):
                with torch.set_grad_enabled(True):
                    anc_img = batch_sample['anc_img'].cuda()
                    pos_img = batch_sample['pos_img'].cuda()
                    neg_img = batch_sample['neg_img'].cuda()

                    with torch.no_grad():
                        features_x_anc, features_x_pos, features_x_neg = model(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                    pos_cls = batch_sample['pos_class'].cuda()
                    neg_cls = batch_sample['neg_class'].cuda()

                    anc_img, _, cover_anc_img = steg.steg_forward(Hnet, cover_dataset, anc_img, 160, opt.device, True, ftype='special')
                    pos_img, _, cover_pos_img = steg.steg_forward(Hnet, cover_dataset, pos_img, 160, opt.device, True, ftype='special')
                    neg_img, _, cover_neg_img = steg.steg_forward(Hnet, cover_dataset, neg_img, 160, opt.device, True, ftype='special')

                    cover_anc_img = cover_anc_img.cuda()
                    cover_pos_img = cover_pos_img.cuda()
                    cover_neg_img = cover_neg_img.cuda()
                    anc_img, pos_img, neg_img = noiser([torch.cat([anc_img, pos_img, neg_img], 0), torch.cat([cover_anc_img, cover_pos_img, cover_neg_img], 0)])[0].chunk(3)

                    anc_lossH = 1 - msssim(anc_img, cover_anc_img, normalize=True)
                    pos_lossH = 1 - msssim(pos_img, cover_pos_img, normalize=True)
                    neg_lossH = 1 - msssim(neg_img, cover_neg_img, normalize=True)

                    anc_embedA, pos_embedA, neg_embedA = modelA(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)
                    anc_embedB, pos_embedB, neg_embedB = modelB(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                    with torch.no_grad():
                        features_y_anc, features_y_pos, features_y_neg = model(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                    distsA = l2_dist.forward(anc_embedA, pos_embedA)
                    distsB = l2_dist.forward(anc_embedB, pos_embedB)
                    distancesA.append(distsA.data.cpu().numpy())
                    distancesB.append(distsB.data.cpu().numpy())
                    labelsA.append(np.ones(distsA.size(0)))
                    labelsB.append(np.ones(distsB.size(0)))

                    distsA = l2_dist.forward(anc_embedA, neg_embedA)
                    distsB = l2_dist.forward(anc_embedB, neg_embedB)
                    distancesA.append(distsA.data.cpu().numpy())
                    distancesB.append(distsB.data.cpu().numpy())
                    labelsA.append(np.zeros(distsA.size(0)))
                    labelsB.append(np.zeros(distsB.size(0)))

                    img_predA = modelA.module.forward_classifier(torch.cat([pos_embedA, neg_embedA], 0))
                    img_predB = modelB.module.forward_classifier(torch.cat([pos_embedB, neg_embedB], 0))

                    true_labels = torch.cat([Variable(pos_cls), Variable(neg_cls)])
                    true_labels = torch.squeeze(true_labels, 1)

                    lossH = (anc_lossH + pos_lossH + neg_lossH) / 3
                    content_loss = (mse_loss(features_x_anc, features_y_anc)\
                                + mse_loss(features_x_pos, features_y_pos)\
                                + mse_loss(features_x_neg, features_y_neg)) / 3
                    criterion = nn.CrossEntropyLoss()
                    cross_entropy_lossA = criterion(img_predA, true_labels) * 0.1
                    cross_entropy_lossB = criterion(img_predB, true_labels) * 0.1
                    triplet_lossA = triplet.forward(anc_embedA, pos_embedA, neg_embedA)
                    triplet_lossB = triplet.forward(anc_embedB, pos_embedB, neg_embedB)
                    lossA = cross_entropy_lossA + 2 * triplet_lossA
                    lossB = cross_entropy_lossB + 2 * triplet_lossB
                    img_predA = nn.Softmax(dim=-1)(img_predA)
                    img_predB = nn.Softmax(dim=-1)(img_predB)
                    lossADV = torch.mean(torch.abs(img_predA-img_predB)) * 1000.

                    lambda1 = (lossADV / (lossA + lossB + lossADV)).item()
                    lambda2 = ((lossA + lossB) / (lossA + lossB + lossADV)).item()
                    
                    loss = 10 * lambda1 * (lossA + lossB) - lambda2 * lossADV
                    loss *= 0.5

                    optimizerH.zero_grad()
                    optimizerD1.zero_grad()
                    optimizerD2.zero_grad()
                    loss.backward()
                    optimizerD1.step()
                    optimizerD2.step()

                    I_cross_entropy_loss_sumA += cross_entropy_lossA.item()
                    I_cross_entropy_loss_sumB += cross_entropy_lossB.item()
                    I_triplet_loss_sumA += triplet_lossA.item()
                    I_triplet_loss_sumB += triplet_lossB.item()
                    I_loss_sumA += lossA.item()
                    I_loss_sumB += lossB.item()
                    I_lossH_sum += lossH.item()
                    I_content_loss_sum += content_loss.item()
                    I_total_loss_sum += loss.item()
                    I_lossADV_sum += lossADV.item()

                    logger.log_value('I cross_entropy_lossA', cross_entropy_lossA.item()).step()
                    logger.log_value('I triplet_lossA', triplet_lossA.item()).step()
                    logger.log_value('I lossA', lossA.item()).step()
                    logger.log_value('I cross_entropy_lossB', cross_entropy_lossB.item()).step()
                    logger.log_value('I triplet_lossB', triplet_lossB.item()).step()
                    logger.log_value('I lossB', lossB.item()).step()
                    logger.log_value('I lossADV', lossADV.item()).step()
                    logger.log_value('I total_loss', loss.item()).step()

            I_avg_cross_entropy_lossA = I_cross_entropy_loss_sumA / (batch_idx+1)
            I_avg_cross_entropy_lossB = I_cross_entropy_loss_sumB / (batch_idx+1)
            I_avg_triplet_lossA = I_triplet_loss_sumA / (batch_idx+1)
            I_avg_triplet_lossB = I_triplet_loss_sumB / (batch_idx+1)
            I_avg_lossA = I_loss_sumA / (batch_idx+1)
            I_avg_lossB = I_loss_sumB / (batch_idx+1)
            I_avg_lossH = I_lossH_sum / (batch_idx+1)
            I_avg_content_loss = I_content_loss_sum / (batch_idx+1)
            I_avg_lossADV = I_lossADV_sum / (batch_idx+1)
            I_avg_total_loss = I_total_loss_sum / (batch_idx+1)

            torch.cuda.synchronize()
            toc = time.time()

            print(' Time = {:.2f} s'.format(toc-tic))
            print(' I  train set - Cross Entropy LossA = {:.8f}'.format(I_avg_cross_entropy_lossA))
            print(' I  train set - Triplet LossA = {:.8f}'.format(I_avg_triplet_lossA))
            print(' I  train set - LossA = {:.8f}'.format(I_avg_lossA))
            print(' I  train set - Cross Entropy LossB = {:.8f}'.format(I_avg_cross_entropy_lossB))
            print(' I  train set - Triplet LossB = {:.8f}'.format(I_avg_triplet_lossB))
            print(' I  train set - LossB = {:.8f}'.format(I_avg_lossB))
            print(' I  train set - LossH = {:.8f}'.format(I_avg_lossH))
            print(' I  train set - content_loss = {:.8f}'.format(I_avg_content_loss))
            print(' I  train set - ADV Loss = {:.8f}'.format(I_avg_lossADV))
            print(' ==> I  train set - Total Loss = {:.8f}'.format(I_avg_total_loss))

            with open(opt.logs_dir + '/train_classifier_log_epoch{}.txt'.format(epoch), 'w') as f:
                f.write(str(epoch) + '\t' +
                        str(I_avg_total_loss))

        if epoch % 2 == 0:
            torch.cuda.synchronize()
            tic = time.time()
            labelsA, distancesA = [], []
            labelsB, distancesB = [], []
            mse_loss_sum = []
            II_total_loss_sum = 0.0
            II_loss_sum = 0.0
            II_lossH_sum = 0.0
            II_cross_entropy_loss_sumA = 0.0
            II_cross_entropy_loss_sumB = 0.0
            II_triplet_loss_sumA = 0.0
            II_triplet_loss_sumB = 0.0
            II_content_loss_sum = 0.0

            for batch_idx, batch_sample in enumerate(data_loaders['train']):
                with torch.set_grad_enabled(True):
                    anc_img = batch_sample['anc_img'].cuda()
                    pos_img = batch_sample['pos_img'].cuda()
                    neg_img = batch_sample['neg_img'].cuda()
                    
                    pos_cls = batch_sample['pos_class'].cuda()
                    neg_cls = batch_sample['neg_class'].cuda()

                    features_x_anc, features_x_neg = model(torch.cat([anc_img, neg_img], 0)).chunk(2)

                    anc_img, _, cover_anc_img = steg.steg_forward(Hnet, cover_dataset, anc_img, 160, opt.device, True, ftype='special')
                    pos_img, _, cover_pos_img = steg.steg_forward(Hnet, cover_dataset, pos_img, 160, opt.device, True, ftype='special')
                    neg_img, _, cover_neg_img = steg.steg_forward(Hnet, cover_dataset, neg_img, 160, opt.device, True, ftype='special')

                    cover_anc_img = cover_anc_img.cuda()
                    cover_pos_img = cover_pos_img.cuda()
                    cover_neg_img = cover_neg_img.cuda()
                    
                    anc_img, pos_img, neg_img = noiser([torch.cat([anc_img, pos_img, neg_img], 0), torch.cat([cover_anc_img, cover_pos_img, cover_neg_img], 0)])[0].chunk(3)

                    anc_lossH = 1 - msssim(anc_img, cover_anc_img, normalize=True) + mse_loss(anc_img, cover_anc_img.to(opt.device))
                    pos_lossH = 1 - msssim(pos_img, cover_pos_img, normalize=True) + mse_loss(pos_img, cover_pos_img.to(opt.device))
                    neg_lossH = 1 - msssim(neg_img, cover_neg_img, normalize=True) + mse_loss(neg_img, cover_neg_img.to(opt.device))

                    anc_embedA, pos_embedA, neg_embedA = modelA(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)
                    anc_embedB, pos_embedB, neg_embedB = modelB(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)
             
                    distsA = l2_dist.forward(anc_embedA, pos_embedA)
                    distsB = l2_dist.forward(anc_embedB, pos_embedB)
                    distancesA.append(distsA.data.cpu().numpy())
                    distancesB.append(distsB.data.cpu().numpy())
                    labelsA.append(np.ones(distsA.size(0)))
                    labelsB.append(np.ones(distsB.size(0)))

                    distsA = l2_dist.forward(anc_embedA, neg_embedA)
                    distsB = l2_dist.forward(anc_embedB, neg_embedB)
                    distancesA.append(distsA.data.cpu().numpy())
                    distancesB.append(distsB.data.cpu().numpy())
                    labelsA.append(np.zeros(distsA.size(0)))
                    labelsB.append(np.zeros(distsB.size(0)))


                    img_predA = modelA.module.forward_classifier(torch.cat([pos_embedA, neg_embedA], 0))
                    img_predB = modelB.module.forward_classifier(torch.cat([pos_embedB, neg_embedB], 0))    

                    features_y_anc, features_y_neg = model(torch.cat([anc_img, neg_img], 0)).chunk(2)
                    
                    true_labels = torch.cat([Variable(pos_cls), Variable(neg_cls)])
                    true_labels = torch.squeeze(true_labels, 1)

                    content_loss = (mse_loss(features_x_anc, features_y_anc)\
                                + mse_loss(features_x_neg, features_y_neg)) / 2
                    img_mse_loss = (mse_loss(anc_img, cover_anc_img) + mse_loss(pos_img, cover_pos_img) + mse_loss(neg_img, cover_neg_img)) / 3
                    lossH = (anc_lossH + pos_lossH + neg_lossH) / 3
                    
                    criterion = nn.CrossEntropyLoss()
                    cross_entropy_lossA = criterion(img_predA, true_labels) * 0.1
                    cross_entropy_lossB = criterion(img_predB, true_labels) * 0.1

                    triplet_lossA = triplet.forward(anc_embedA, pos_embedA, neg_embedA)
                    triplet_lossB = triplet.forward(anc_embedB, pos_embedB, neg_embedB)
                    img_predA = nn.Softmax(dim=-1)(img_predA)
                    img_predB = nn.Softmax(dim=-1)(img_predB)
                    loss = torch.mean(torch.abs(img_predA-img_predB)) * 1000.
                    
                    lossA = cross_entropy_lossA + 2 * triplet_lossA
                    lossB = cross_entropy_lossB + 2 * triplet_lossB
                    
                    lambda1 = ((lossH + lossA + lossB + content_loss) / (loss + lossH + content_loss + lossA + lossB)).item()
                    lambda2 = ((loss + lossA + lossB + content_loss) / (loss + lossH + content_loss + lossA + lossB)).item()
                    lambda3 = ((loss + lossH + lossA + lossB) / (loss + lossH + content_loss + lossA + lossB)).item()
                    lambda4 = ((lossH + loss + content_loss) / (loss + lossH + content_loss + lossA + lossB)).item()
                    
                    if save_lossH.avg is None or save_lossH.avg > 0.005:
                        total_loss = 5 * lambda1 * loss + 130 * lambda2 * lossH + 2 * lambda3 * content_loss + 0.1 * lambda4 * (lossA + lossB)
                    else:
                        total_loss = 5 * lambda1 * loss + 10 * lambda2 * lossH + 2 * lambda3 * content_loss + 0.1 * lambda4 * (lossA + lossB)

                    total_loss *= 0.5
                    cur_lossH.update(lossH.item())

                    optimizerH.zero_grad()
                    optimizerD1.zero_grad()
                    optimizerD2.zero_grad()
                    total_loss.backward()
                    optimizerH.step()
                    

                    II_lossH_sum += lossH.item()
                    II_loss_sum += loss.item() 
                    II_total_loss_sum += total_loss.item()
                    II_content_loss_sum += content_loss.item()
                    mse_loss_sum.append(img_mse_loss.item())

                    II_cross_entropy_loss_sumA += cross_entropy_lossA.item()
                    II_cross_entropy_loss_sumB += cross_entropy_lossB.item()
                    
                    II_triplet_loss_sumA += triplet_lossA.item()
                    II_triplet_loss_sumB += triplet_lossB.item()

                    logger.log_value('II ADV Loss', loss.item()).step()
                    logger.log_value('II lossH', lossH.item()).step()
                    logger.log_value('II total_loss', total_loss.item()).step()

            II_avg_lossH = II_lossH_sum / (batch_idx+1)
            II_avg_adv_loss = II_loss_sum / (batch_idx+1)
            II_avg_total_loss = II_total_loss_sum / (batch_idx+1)
            II_avg_content_loss = II_content_loss_sum / (batch_idx+1)
            II_avg_cross_entropy_lossA = II_cross_entropy_loss_sumA / (batch_idx+1)
            II_avg_cross_entropy_lossB = II_cross_entropy_loss_sumB / (batch_idx+1)
            II_avg_triplet_lossA = II_triplet_loss_sumA / (batch_idx+1)
            II_avg_triplet_lossB = II_triplet_loss_sumB / (batch_idx+1)
            avg_mse_loss = np.mean(mse_loss_sum)
            psnr = -10 * np.log10(avg_mse_loss)
            
            save_lossH = copy.deepcopy(cur_lossH)
            cur_lossH.reset()

            torch.cuda.synchronize()
            toc = time.time()

            print(' Time = {:.2f} s'.format(toc-tic))
            print(' II  train set - ADV Loss = {:.8f}'.format(II_avg_adv_loss))
            print(' II  train set - LossH = {:.8f}'.format(II_avg_lossH))
            print(' II  train set - cross_entropy_lossA = {:.8f}'.format(np.mean(II_avg_cross_entropy_lossA)))
            print(' II  train set - cross_entropy_lossB = {:.8f}'.format(np.mean(II_avg_cross_entropy_lossB)))
            print(' II  train set - triplet_lossA = {:.8f}'.format(np.mean(II_avg_triplet_lossA)))
            print(' II  train set - triplet_lossB = {:.8f}'.format(np.mean(II_avg_triplet_lossB)))
            print(' II  train set - content_loss = {:.8f}'.format(II_avg_content_loss))
            print(' ==> II  train set - total_Loss = {:.8f}'.format(II_avg_total_loss))
            print(' II  train set - PSNR = {:.3f}'.format(psnr))

            with open(opt.logs_dir + '/classifier_log_epoch{}.txt'.format(epoch), 'w') as f:
                f.write(str(epoch) + '\t' +
                        str(II_avg_total_loss))
        
        vutils.save_image(cover_anc_img, "visualize/cover_image.jpg", nrow=1, padding=0, normalize=False)
        vutils.save_image(anc_img, "visualize/container_image.jpg", nrow=1, padding=0, normalize=False)
                               
        if epoch % 5 == 0:
            Hnet.eval()
            model.eval()
            modelA.eval()
            modelB.eval()
            noiser.eval()
            print(80 * '=')
            labelsA, distancesA = [], []
            labelsB, distancesB = [], []
            cross_entropy_loss_sumA = []
            cross_entropy_loss_sumB = []
            triplet_loss_sumA = []
            triplet_loss_sumB = []
            total_loss_sum = []
            mse_loss_sum = []
            lossH_sum = []
            content_loss_sum = []

            l2_dist = PairwiseDistance(2)

            for batch_idx, batch_sample in enumerate(data_loaders['valid']):
                with torch.no_grad():
                    anc_img = batch_sample['anc_img'].cuda()
                    pos_img = batch_sample['pos_img'].cuda()
                    neg_img = batch_sample['neg_img'].cuda()

                    pos_cls = batch_sample['pos_class'].cuda()
                    neg_cls = batch_sample['neg_class'].cuda()

                    # TODO
                    features_x_anc, features_x_pos, features_x_neg = model(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                    anc_img, anc_lossH, cover_anc_img = steg.steg_forward(Hnet, cover_dataset, anc_img, 160,
                                                           opt.device,
                                                           True, ftype='special')
                    pos_img, pos_lossH, cover_pos_img = steg.steg_forward(Hnet, cover_dataset, pos_img, 160,
                                                           opt.device,
                                                           True, ftype='special')
                    neg_img, neg_lossH, cover_neg_img = steg.steg_forward(Hnet, cover_dataset, neg_img, 160,
                                                           opt.device,
                                                           True, ftype='special')
                    
                    anc_embedA, pos_embedA, neg_embedA = modelA(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)
                    anc_embedB, pos_embedB, neg_embedB = modelB(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                    distsA = l2_dist.forward(anc_embedA, pos_embedA)
                    distancesA.append(distsA.data.cpu().numpy())
                    labelsA.append(np.ones(distsA.size(0)))
                    distsB = l2_dist.forward(anc_embedB, pos_embedB)
                    distancesB.append(distsB.data.cpu().numpy())
                    labelsB.append(np.ones(distsB.size(0)))

                    distsA = l2_dist.forward(anc_embedA, neg_embedA)
                    distancesA.append(distsA.data.cpu().numpy())
                    labelsA.append(np.zeros(distsA.size(0)))
                    distsB = l2_dist.forward(anc_embedB, neg_embedB)
                    distancesB.append(distsB.data.cpu().numpy())
                    labelsB.append(np.zeros(distsB.size(0)))
                    
                    features_y_anc, features_y_pos, features_y_neg = model(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                    img_predA = modelA.module.forward_classifier(torch.cat([pos_embedA, neg_embedA], 0))
                    img_predB = modelB.module.forward_classifier(torch.cat([pos_embedB, neg_embedB], 0))

                    true_labels = torch.cat([Variable(pos_cls), Variable(neg_cls)])
                    true_labels = torch.squeeze(true_labels, 1)

                    lossH = (anc_lossH + pos_lossH + neg_lossH) / 3
                    criterion = nn.CrossEntropyLoss()
                    cross_entropy_lossA = criterion(img_predA, true_labels)
                    cross_entropy_lossB = criterion(img_predB, true_labels)
                    triplet_lossA = triplet.forward(anc_embedA, pos_embedA, neg_embedA)
                    triplet_lossB = triplet.forward(anc_embedB, pos_embedB, neg_embedB)

                    img_mse_loss = (mse_loss(anc_img, cover_anc_img) + mse_loss(pos_img, cover_pos_img) + mse_loss(neg_img, cover_neg_img)) / 3
                    content_loss = (mse_loss(features_x_anc, features_y_anc)\
                                + mse_loss(features_x_pos, features_y_pos)\
                                + mse_loss(features_x_neg, features_y_neg)) / 3

                    lossA = cross_entropy_lossA + 1.5 * triplet_lossA
                    lossB = cross_entropy_lossB + 1.5 * triplet_lossB
                    total_loss = lossA + lossB + lossH

                    cross_entropy_loss_sumA.append(cross_entropy_lossA.item())
                    cross_entropy_loss_sumB.append(cross_entropy_lossB.item())
                    triplet_loss_sumA.append(triplet_lossA.item())
                    triplet_loss_sumB.append(triplet_lossB.item())
                    lossH_sum.append(lossH.item())
                    total_loss_sum.append(total_loss.item())
                    content_loss_sum.append(content_loss.item())
                    mse_loss_sum.append(img_mse_loss.item())

            avg_cross_entropy_lossA = np.mean(cross_entropy_loss_sumA)
            avg_triplet_lossA = np.mean(triplet_loss_sumA)
            avg_cross_entropy_lossB = np.mean(cross_entropy_loss_sumB)
            avg_triplet_lossB = np.mean(triplet_loss_sumB)
            avg_lossH = np.mean(lossH_sum)
            avg_content_loss = np.mean(content_loss_sum)
            avg_total_loss = np.mean(total_loss_sum)
            avg_mse_loss = np.mean(mse_loss_sum)
            psnr = -10 * np.log10(avg_mse_loss)


            labels = np.array([sublabel for label in labelsA for sublabel in label])
            distances = np.array([subdist for dist in distancesA for subdist in dist])

            tpr, fpr, accuracy1, val, val_std, far = evaluate(distances, labels)

            labels = np.array([sublabel for label in labelsB for sublabel in label])
            distances = np.array([subdist for dist in distancesB for subdist in dist])

            tpr, fpr, accuracy2, val, val_std, far = evaluate(distances, labels)

            print('  valid set - Cross Entropy LossA = {:.8f}'.format(avg_cross_entropy_lossA))
            print('  valid set - Triplet LossA = {:.8f}'.format(avg_triplet_lossA))
            print('  valid set - Accuracy1 = {:.8f}'.format(np.mean(accuracy1)))
            print('\n  valid set - Cross Entropy LossB = {:.8f}'.format(avg_cross_entropy_lossB))
            print('  valid set - Triplet LossB = {:.8f}'.format(avg_triplet_lossB))
            print('  valid set - Accuracy2 = {:.8f}'.format(np.mean(accuracy2)))
            print(' ==>  valid set - LossH = {:.8f}'.format(avg_lossH))
            print(' ==>  valid set - content loss = {:.8f}'.format(avg_content_loss))
            print(' ==>  valid set - Total Loss = {:.8f}'.format(avg_total_loss))
            print(' ==>  valid set - PSNR = {:.3f}'.format(psnr))

            logger.log_value('valid_cross_entropy_lossA', avg_cross_entropy_lossA).step()
            logger.log_value('valid_triplet_lossA', avg_triplet_lossA).step()
            logger.log_value('valid_accuracy1', np.mean(accuracy1)).step()
            logger.log_value('valid_cross_entropy_lossB', avg_cross_entropy_lossB).step()
            logger.log_value('valid_triplet_lossB', avg_triplet_lossB).step()
            logger.log_value('valid_accuracy2', np.mean(accuracy2)).step()
            logger.log_value('valid_lossH', avg_lossH).step()
            logger.log_value('valid_psnr', psnr).step()
            logger.log_value('valid_total_loss', avg_total_loss).step()
        if epoch % 5 == 0:
            print("save at", opt.classifier_ckpt + '/checkpoint_epoch{}.pth'.format(epoch))
            torch.save({'epoch': epoch,
                        'Hnet_state_dict': Hnet.state_dict(),
                        'clsA_state_dict': modelA.state_dict(),
                        'clsB_state_dict': modelB.state_dict()},
                       opt.classifier_ckpt + '/checkpoint_epoch{}.pth'.format(epoch))


def test(**kwargs):

    opt.load_steg_model_dir = './model_zoo/steg_simplify_high_acc_model.pth'
    opt.load_classifier_model_dir1 = './model_zoo/steg_simplify_high_acc_model.pth'
    opt.load_classifier_model_dir2 = './model_zoo/steg_simplify_high_acc_model.pth'
    
    opt.parse(kwargs)
    opt.device = 'cuda:0'
    print(opt.device)
    
    NoiseArgParser()(opt.noise, opt)
    
    modelA = FaceNetInceptionModel(opt.embedding_size, opt.num_classes)
    modelA = modelA.cuda()
    modelB = FaceNetInceptionModel(opt.embedding_size, opt.num_classes)
    modelB = modelB.cuda()
    noiser = Noiser(opt.noise, opt.device)
    noiser = noiser.cuda()

    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Hnet = Hnet.cuda()
    
    cover_dataset = MyImageFolder(
        opt.cover_dir,
        transforms.Compose([
            transforms.Resize([opt.image_size, opt.image_size]),
            transforms.ToTensor(),
        ]))


    if opt.load_steg_model_dir != '':
        checkpoint = torch.load(opt.load_steg_model_dir, map_location={'cuda:1':opt.device})
        Hnet.load_state_dict(preprocess_ckpt(checkpoint['Hnet_state_dict']))

    if opt.load_classifier_model_dir1 != '':
        checkpoint = torch.load(opt.load_classifier_model_dir1, map_location={'cuda:1':opt.device})
        modelA.load_state_dict(preprocess_ckpt(checkpoint['clsA_state_dict']))

    if opt.load_classifier_model_dir2 != '':
        checkpoint = torch.load(opt.load_classifier_model_dir2, map_location={'cuda:1':opt.device})
        modelB.load_state_dict(preprocess_ckpt(checkpoint['clsB_state_dict']))

    modelA = DataParallel(modelA)
    modelB = DataParallel(modelB)
    noiser = DataParallel(noiser)
    Hnet = DataParallel(Hnet)

    mse_loss = torch.nn.MSELoss()
    l2_dist = PairwiseDistance(2)

    data_loaders, data_size = get_dataloader(opt.train_root_dir, opt.valid_root_dir,
                                                opt.train_csv_name, opt.valid_csv_name,
                                                opt.num_train_triplets, opt.num_valid_triplets,
                                                opt.batch_size, opt.num_workers)

    Hnet.eval()
    modelA.eval()
    modelB.eval()
    noiser.eval()
    labelsA, distancesA = [], []
    labelsB, distancesB = [], []
    mse_loss_sum = []

    l2_dist = PairwiseDistance(2)

    for batch_idx, batch_sample in enumerate(data_loaders['valid']):
        with torch.no_grad():
            anc_img = batch_sample['anc_img'].cuda()
            pos_img = batch_sample['pos_img'].cuda()
            neg_img = batch_sample['neg_img'].cuda()

            pos_cls = batch_sample['pos_class'].cuda()
            neg_cls = batch_sample['neg_class'].cuda()

            anc_img, anc_lossH, cover_anc_img = steg.steg_forward(Hnet, cover_dataset, anc_img, 160,
                                                    opt.device,
                                                    True, ftype='special')
            pos_img, pos_lossH, cover_pos_img = steg.steg_forward(Hnet, cover_dataset, pos_img, 160,
                                                    opt.device,
                                                    True, ftype='special')
            neg_img, neg_lossH, cover_neg_img = steg.steg_forward(Hnet, cover_dataset, neg_img, 160,
                                                    opt.device,
                                                    True, ftype='special')
            
            anc_embedA, pos_embedA, neg_embedA = modelA(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)
            anc_embedB, pos_embedB, neg_embedB = modelB(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

            distsA = l2_dist.forward(anc_embedA, pos_embedA)
            distancesA.append(distsA.data.cpu().numpy())
            labelsA.append(np.ones(distsA.size(0)))
            distsB = l2_dist.forward(anc_embedB, pos_embedB)
            distancesB.append(distsB.data.cpu().numpy())
            labelsB.append(np.ones(distsB.size(0)))

            distsA = l2_dist.forward(anc_embedA, neg_embedA)
            distancesA.append(distsA.data.cpu().numpy())
            labelsA.append(np.zeros(distsA.size(0)))
            distsB = l2_dist.forward(anc_embedB, neg_embedB)
            distancesB.append(distsB.data.cpu().numpy())
            labelsB.append(np.zeros(distsB.size(0)))

            true_labels = torch.cat([Variable(pos_cls), Variable(neg_cls)])
            true_labels = torch.squeeze(true_labels, 1)

            img_mse_loss = (mse_loss(anc_img, cover_anc_img) + mse_loss(pos_img, cover_pos_img) + mse_loss(neg_img, cover_neg_img)) / 3

            mse_loss_sum.append(img_mse_loss.item())

    avg_mse_loss = np.mean(mse_loss_sum)
    psnr = -10 * np.log10(avg_mse_loss)

    labels = np.array([sublabel for label in labelsA for sublabel in label])
    distances = np.array([subdist for dist in distancesA for subdist in dist])

    tpr, fpr, accuracy1, val, val_std, far = evaluate(distances, labels)

    labels = np.array([sublabel for label in labelsB for sublabel in label])
    distances = np.array([subdist for dist in distancesB for subdist in dist])

    tpr, fpr, accuracy2, val, val_std, far = evaluate(distances, labels)

    print('  valid set - Accuracy1 = {:.8f}'.format(np.mean(accuracy1)))
    print('  valid set - Accuracy2 = {:.8f}'.format(np.mean(accuracy2)))
    print(' ==>  valid set - PSNR = {:.3f}'.format(psnr))


def visualize(output_path='outputs', **kwargs):
    
    try:
        os.mkdir(output_path)
    except Exception:
        pass

    opt.load_steg_model_dir = './model_zoo/steg_simplify_high_psnr_model.pth'
    
    opt.parse(kwargs)
    opt.device = 'cuda:0'
    print(opt.device)
    
    NoiseArgParser()(opt.noise, opt)
   
    noiser = Noiser(opt.noise, opt.device)
    noiser = noiser.cuda()

    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Hnet = Hnet.cuda()
    
    cover_dataset = MyImageFolder(
        opt.cover_dir,
        transforms.Compose([
            transforms.Resize([opt.image_size, opt.image_size]),
            transforms.ToTensor(),
        ]))

    if opt.load_steg_model_dir != '':
        checkpoint = torch.load(opt.load_steg_model_dir, map_location={'cuda:1':opt.device})
        Hnet.load_state_dict(preprocess_ckpt(checkpoint['Hnet_state_dict']))

    noiser = DataParallel(noiser)
    Hnet = DataParallel(Hnet)

    data_loaders, data_size = get_dataloader(opt.train_root_dir, opt.valid_root_dir, opt.train_csv_name, opt.valid_csv_name, opt.num_train_triplets, opt.num_valid_triplets, opt.batch_size, opt.num_workers, False)

    Hnet.eval()
    noiser.eval()

    for batch_idx, batch_sample in enumerate(data_loaders['valid']):
        with torch.no_grad():
            anc_img = batch_sample['anc_img'].cuda()

            _anc_img, anc_lossH, cover_anc_img = steg.steg_forward(Hnet, cover_dataset, anc_img, 160,
                                                    opt.device,
                                                    True, ftype='special')
        
        vutils.save_image(cover_anc_img, os.path.join(output_path, "cover_img.jpg"), nrow=1, padding=0, normalize=False)
        vutils.save_image(_anc_img, os.path.join(output_path, "container_img.jpg"), nrow=1, padding=0, normalize=False)
        vutils.save_image(anc_img, os.path.join(output_path, "secret_img.jpg"), nrow=1, padding=0, normalize=False)

        break


if __name__ == '__main__':
    import fire
    fire.Fire()