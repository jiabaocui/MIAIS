import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.modules.distance import PairwiseDistance
from torch.nn.parallel.data_parallel import DataParallel
from torchvision import transforms

sys.path.append(os.path.abspath('./steg'))
import steg
from models import UnetGenerator
from models import oriRevealNet
from models import FaceNetInceptionModel
from data import MyImageFolder
from data import get_dataloader
from utils import Logger
from utils import TripletLoss
from utils import AverageMeter
from utils import evaluate, preprocess_ckpt
from utils.AFD_config import opt
from noise_argparser import NoiseArgParser
from noise_layers.noiser import Noiser


def train(**kwargs):

    print('GPUS: ', torch.cuda.device_count())

    opt.num_valid_triplets = 1000
    opt.batch_size = 4

    opt.learning_rate = 1e-4

    opt.load_Hnet_dir = './model_zoo/steg_full_joint_model.pth'
    opt.load_Rnet_dir = './model_zoo/steg_full_joint_model.pth'
    opt.load_classifier_model_dir1 = './model_zoo/steg_full_joint_model.pth'
    opt.load_classifier_model_dir2 = './model_zoo/steg_full_joint_model.pth'
    opt.load_classifier_model_dir = './model_zoo/steg_base_model.pth'

    opt.output_dir = 'visualize'
    opt.root_dir = './results/'
    opt.ckpt_dir = opt.root_dir + 'ckpt'
    opt.logs_dir = opt.root_dir + 'logs'

    if not os.path.exists(opt.root_dir):
        os.mkdir(opt.root_dir)
    if not os.path.exists(opt.ckpt_dir):
        os.mkdir(opt.ckpt_dir)
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    opt.device = 'cuda:0'

    opt.is_test = False
    opt.is_valid = False

    # ***************************************************************************************** #

    modelA = FaceNetInceptionModel(opt.embedding_size, opt.num_classes, pretrained=False)
    modelA.cuda()
    modelB = FaceNetInceptionModel(opt.embedding_size, opt.num_classes, pretrained=False)
    modelB.cuda()
    NoiseArgParser()(opt.noise, opt)
    noiser = Noiser(opt.noise, opt.device)
    noiser.cuda()
    model = FaceNetInceptionModel(opt.embedding_size, opt.num_classes, pretrained=False)
    model.cuda()
    model.eval()

    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Hnet.cuda()
    Rnet = oriRevealNet()
    Rnet.cuda()

    if opt.load_Hnet_dir != '':
        checkpoint = torch.load(opt.load_Hnet_dir, map_location='cpu')
        Hnet.load_state_dict(checkpoint['Hnet_state_dict'])

    if opt.load_Rnet_dir != '':
        checkpoint = torch.load(opt.load_Rnet_dir, map_location='cpu')
        Rnet.load_state_dict(checkpoint['Rnet_state_dict'])

    if opt.load_classifier_model_dir1 != '':
        checkpoint = torch.load(opt.load_classifier_model_dir1, map_location='cpu')
        modelA.load_state_dict(checkpoint['cls_state_dict'])

    if opt.load_classifier_model_dir2 != '':
        checkpoint = torch.load(opt.load_classifier_model_dir2, map_location='cpu')
        modelB.load_state_dict(checkpoint['cls_state_dict'])

    if opt.load_classifier_model_dir != '':
        checkpoint = torch.load(opt.load_classifier_model_dir, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

    modelA = DataParallel(modelA)
    modelB = DataParallel(modelB)
    model = DataParallel(model)
    noiser = DataParallel(noiser)
    Hnet = DataParallel(Hnet)
    Rnet = DataParallel(Rnet)

    optimizerH = optim.SGD(Hnet.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizerR = optim.SGD(Rnet.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizerD1 = optim.SGD(modelA.parameters(), lr=opt.learning_rate * 0.1, momentum=0.9, weight_decay=5e-4)
    optimizerD2 = optim.SGD(modelB.parameters(), lr=opt.learning_rate * 0.1, momentum=0.9, weight_decay=5e-4)

    train_cover_dataset = MyImageFolder(
        opt.train_cover_dir,
        transforms.Compose([
            transforms.Resize([opt.image_size, opt.image_size]),
            transforms.ToTensor(),
        ]))

    for param in model.parameters():
        param.requires_grad = False

    if opt.start_epoch == 0:
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
        Rnet.train()
        noiser.train()
        data_loaders, data_size = get_dataloader(opt.train_root_dir, opt.valid_root_dir,
                                                 opt.train_csv_name, opt.valid_csv_name,
                                                 opt.num_train_triplets, opt.num_valid_triplets,
                                                 opt.batch_size, opt.num_workers, False)

        if epoch % 2 == 1:
            labelsA, distancesA = [], []
            labelsB, distancesB = [], []
            I_cross_entropy_loss_sumA = 0.0
            I_cross_entropy_loss_sumB = 0.0
            I_triplet_loss_sumA = 0.0
            I_triplet_loss_sumB = 0.0
            I_loss_sumA = 0.0
            I_loss_sumB = 0.0
            I_lossH_sum = 0.0
            I_lossR_sum = 0.0
            I_lossADV_sum = 0.0
            I_content_loss_sum = 0.0
            I_total_loss_sum = 0.0

            for batch_idx, batch_sample in enumerate(data_loaders['train']):
                if opt.is_test:
                    if batch_idx == 5:
                        break
                if opt.is_valid:
                    break

                with torch.set_grad_enabled(True):
                    anc_img = batch_sample['anc_img'].cuda()
                    pos_img = batch_sample['pos_img'].cuda()
                    neg_img = batch_sample['neg_img'].cuda()

                    _anc_img = anc_img
                    _pos_img = pos_img
                    _neg_img = neg_img

                    features_x_anc, features_x_pos, features_x_neg = model(
                        torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                    pos_cls = batch_sample['pos_class'].cuda()
                    neg_cls = batch_sample['neg_class'].cuda()

                    anc_img, anc_lossH, _, cover_anc_img = steg.steg_forward_vggm(Hnet, train_cover_dataset,
                                                                                  anc_img, None,
                                                                                  None, opt, True)
                    pos_img, pos_lossH, _, cover_pos_img = steg.steg_forward_vggm(Hnet, train_cover_dataset,
                                                                                  pos_img, None,
                                                                                  None, opt, True)
                    neg_img, neg_lossH, _, cover_neg_img = steg.steg_forward_vggm(Hnet, train_cover_dataset,
                                                                                  neg_img, None,
                                                                                  None, opt, True)

                    anc_embedA, pos_embedA, neg_embedA = modelA(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(
                        3)
                    anc_embedB, pos_embedB, neg_embedB = modelB(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(
                        3)

                    source_pos_img_predA = modelA.module.forward_classifier(pos_embedA).cuda()
                    source_neg_img_predA = modelA.module.forward_classifier(neg_embedA).cuda()

                    source_pos_img_predB = modelB.module.forward_classifier(pos_embedB).cuda()
                    source_neg_img_predB = modelB.module.forward_classifier(neg_embedB).cuda()

                    anc_embedA, pos_embedA, neg_embedA = modelA(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(
                        3)
                    anc_embedB, pos_embedB, neg_embedB = modelB(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(
                        3)

                    features_y_anc, features_y_pos, features_y_neg = \
                        model(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

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

                    pos_img_predA, neg_img_predA = modelA.module.forward_classifier(
                        torch.cat([pos_embedA, neg_embedA], 0)).cuda().chunk(2)
                    pos_img_predB, neg_img_predB = modelB.module.forward_classifier(
                        torch.cat([pos_embedB, neg_embedB], 0)).cuda().chunk(2)

                    img_predA = torch.cat([pos_img_predA, neg_img_predA])
                    img_predB = torch.cat([pos_img_predB, neg_img_predB])

                    source_img_predA = torch.cat([source_pos_img_predA, source_neg_img_predA])
                    source_img_predB = torch.cat([source_pos_img_predB, source_neg_img_predB])

                    true_labels = torch.cat([Variable(pos_cls.cuda()), Variable(neg_cls.cuda())])
                    true_labels = torch.squeeze(true_labels, 1)

                    content_loss = (mse_loss(features_x_anc, features_y_anc) \
                                    + mse_loss(features_x_pos, features_y_pos) \
                                    + mse_loss(features_x_neg, features_y_neg)) / 3
                    lossH = (anc_lossH + pos_lossH + neg_lossH) / 3

                    anc_img_rev, anc_lossR = steg.reveal_forward_vgg(Rnet, _anc_img, anc_img, opt, True)
                    pos_img_rev, pos_lossR = steg.reveal_forward_vgg(Rnet, _pos_img, pos_img, opt, True)
                    neg_img_rev, neg_lossR = steg.reveal_forward_vgg(Rnet, _neg_img, neg_img, opt, True)

                    lossR = (anc_lossR + pos_lossR + neg_lossR) / 3

                    criterion = nn.CrossEntropyLoss()
                    cross_entropy_lossA = criterion(source_img_predA.cuda(), true_labels.cuda())
                    cross_entropy_lossB = criterion(source_img_predB.cuda(), true_labels.cuda())
                    triplet_lossA = TripletLoss(opt.margin).forward(anc_embedA, pos_embedA, neg_embedA).cuda()
                    triplet_lossB = TripletLoss(opt.margin).forward(anc_embedB, pos_embedB, neg_embedB).cuda()


                    lossA = 1 * cross_entropy_lossA + 2 * triplet_lossA
                    lossB = 1 * cross_entropy_lossB + 2 * triplet_lossB

                    img_predA = nn.Softmax(dim=-1)(img_predA)
                    img_predB = nn.Softmax(dim=-1)(img_predB)
                    lossADV = torch.mean(torch.abs(img_predA - img_predB)) * 1000

                    lambda1 = (lossADV / (lossA + lossB + lossADV)).item()
                    lambda2 = ((lossA + lossB) / (lossA + lossB + lossADV)).item()

                    loss = 15 * lambda1 * (lossA + lossB) - 1.5 * lambda2 * lossADV

                    if batch_idx % 8 == 0:
                        optimizerH.zero_grad()
                        optimizerR.zero_grad()
                        optimizerD1.zero_grad()
                        optimizerD2.zero_grad()
                    loss.backward()
                    if batch_idx % 8 == 7:
                        optimizerD1.step()
                        optimizerD2.step()

                    I_cross_entropy_loss_sumA += cross_entropy_lossA.item()
                    I_cross_entropy_loss_sumB += cross_entropy_lossB.item()
                    I_triplet_loss_sumA += triplet_lossA.item()
                    I_triplet_loss_sumB += triplet_lossB.item()
                    I_loss_sumA += lossA.item()
                    I_loss_sumB += lossB.item()
                    I_lossH_sum += lossH.item()
                    I_lossR_sum += lossR.item()
                    I_content_loss_sum += content_loss.item()
                    I_total_loss_sum += loss.item()
                    I_lossADV_sum += lossADV.item()

                    del anc_img, cover_anc_img, anc_img_rev
                    del pos_img, cover_pos_img, pos_img_rev
                    del neg_img, cover_neg_img, neg_img_rev
                    del features_x_anc, features_x_pos, features_x_neg
                    del features_y_anc, features_y_pos, features_y_neg
                    torch.cuda.empty_cache()

            if opt.is_valid == False:
                I_avg_cross_entropy_lossA = I_cross_entropy_loss_sumA / (batch_idx + 1)
                I_avg_cross_entropy_lossB = I_cross_entropy_loss_sumB / (batch_idx + 1)
                I_avg_triplet_lossA = I_triplet_loss_sumA / (batch_idx + 1)
                I_avg_triplet_lossB = I_triplet_loss_sumB / (batch_idx + 1)
                I_avg_lossA = I_loss_sumA / (batch_idx + 1)
                I_avg_lossB = I_loss_sumB / (batch_idx + 1)
                I_avg_lossH = I_lossH_sum / (batch_idx + 1)
                I_avg_lossR = I_lossR_sum / (batch_idx + 1)
                I_avg_content_loss = I_content_loss_sum / (batch_idx + 1)
                I_avg_lossADV = I_lossADV_sum / (batch_idx + 1)
                I_avg_total_loss = I_total_loss_sum / (batch_idx + 1)

                logger.log_value('I cross_entropy_lossA', I_avg_cross_entropy_lossA).step()
                logger.log_value('I triplet_lossA', I_avg_triplet_lossA).step()
                logger.log_value('I lossA', I_avg_lossA).step()
                logger.log_value('I cross_entropy_lossB', I_avg_cross_entropy_lossB).step()
                logger.log_value('I triplet_lossB', I_avg_triplet_lossB).step()
                logger.log_value('I lossB', I_avg_lossB).step()
                logger.log_value('I lossADV', I_avg_lossADV).step()
                logger.log_value('I lossH', I_avg_lossH).step()
                logger.log_value('I lossR', I_avg_lossR).step()
                logger.log_value('I content_loss', I_avg_content_loss).step()
                logger.log_value('I total_loss', I_avg_total_loss).step()

                labels = np.array([sublabel for label in labelsA for sublabel in label])
                distances = np.array([subdist for dist in distancesA for subdist in dist])

                tpr, fpr, accuracy1, val, val_std, far = evaluate(distances, labels)

                labels = np.array([sublabel for label in labelsB for sublabel in label])
                distances = np.array([subdist for dist in distancesB for subdist in dist])

                tpr, fpr, accuracy2, val, val_std, far = evaluate(distances, labels)

                print(80 * '=')
                print('Epoch [{}/{}]'.format(epoch, opt.num_epochs + opt.start_epoch - 1))
                # print("learingingH rate: {}".format(optimizerH.param_groups[0]['lr']))
                # print("learingingR rate: {}".format(optimizerR.param_groups[0]['lr']))
                # print("learingingD1 rate: {}".format(optimizerD1.param_groups[0]['lr']))
                # print("learingingD2 rate: {}".format(optimizerD2.param_groups[0]['lr']))

                print(' I  train set - Cross Entropy LossA = {:.8f}'.format(I_avg_cross_entropy_lossA))
                print(' I  train set - Cross Entropy LossB = {:.8f}'.format(I_avg_cross_entropy_lossB))
                print(' I  train set - Triplet LossA = {:.8f}'.format(I_avg_triplet_lossA))
                print(' I  train set - Triplet LossB = {:.8f}'.format(I_avg_triplet_lossB))
                print(' I  train set - LossA = {:.8f}'.format(I_avg_lossA))
                print(' I  train set - LossB = {:.8f}'.format(I_avg_lossB))
                print(' train set - AccuracyA = {:.8f}'.format(np.mean(accuracy1)))
                print(' train set - AccuracyB = {:.8f}'.format(np.mean(accuracy2)))
                print(' I  train set - LossH = {:.8f}'.format(I_avg_lossH))
                print(' I  train set - LossR = {:.8f}'.format(I_avg_lossR))
                print(' I  train set - content_loss = {:.8f}'.format(I_avg_content_loss))
                print(' I  train set - ADV Loss = {:.8f}'.format(I_avg_lossADV))
                print(' ==> I  train set - Total Loss = {:.8f}'.format(I_avg_total_loss))

                with open(opt.logs_dir + '/train_classifier_log_epoch{}.txt'.format(epoch), 'w') as f:
                    f.write(str(epoch) + '\t' +
                            str(np.mean(accuracy1)) + '\t' +
                            str(np.mean(accuracy2)) + '\t' +
                            str(I_avg_total_loss))

        if epoch % 2 == 0:
            labelsA, distancesA = [], []
            labelsB, distancesB = [], []
            II_total_loss_sum = 0.0
            II_loss_sum = 0.0
            II_lossH_sum = 0.0
            II_lossR_sum = 0.0
            II_triplet_loss_sumA = 0.0
            II_triplet_loss_sumB = 0.0
            II_content_loss_sum = 0.0

            for batch_idx, batch_sample in enumerate(data_loaders['train']):
                if opt.is_test:
                    if batch_idx == 5:
                        break
                if opt.is_valid:
                    break
                with torch.set_grad_enabled(True):
                    anc_img = batch_sample['anc_img'].cuda()
                    pos_img = batch_sample['pos_img'].cuda()
                    neg_img = batch_sample['neg_img'].cuda()

                    _anc_img = anc_img
                    _pos_img = pos_img
                    _neg_img = neg_img

                    features_x_anc, features_x_pos, features_x_neg = \
                        model(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                    anc_img, anc_lossH, _, cover_anc_img = steg.steg_forward_vggm(Hnet, train_cover_dataset,
                                                                                  anc_img, None,
                                                                                  None, opt, True)
                    pos_img, pos_lossH, _, cover_pos_img = steg.steg_forward_vggm(Hnet, train_cover_dataset,
                                                                                  pos_img, None,
                                                                                  None, opt, True)
                    neg_img, neg_lossH, _, cover_neg_img = steg.steg_forward_vggm(Hnet, train_cover_dataset,
                                                                                  neg_img, None,
                                                                                  None, opt, True)

                    anc_embedA, pos_embedA, neg_embedA = modelA(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(
                        3)
                    anc_embedB, pos_embedB, neg_embedB = modelB(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(
                        3)

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

                    pos_img_predA, neg_img_predA = modelA.module.forward_classifier(
                        torch.cat([pos_embedA, neg_embedA], 0)).cuda().chunk(2)
                    pos_img_predB, neg_img_predB = modelB.module.forward_classifier(
                        torch.cat([pos_embedB, neg_embedB], 0)).cuda().chunk(2)

                    features_y_anc, features_y_pos, features_y_neg = model(
                        torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                    img_predA = torch.cat([pos_img_predA, neg_img_predA])
                    img_predB = torch.cat([pos_img_predB, neg_img_predB])
                    content_loss = (mse_loss(features_x_anc, features_y_anc) \
                                    + mse_loss(features_x_pos, features_y_pos) \
                                    + mse_loss(features_x_neg, features_y_neg)) / 3
                    lossH = (anc_lossH + pos_lossH + neg_lossH) / 3

                    anc_img_rev, anc_lossR = steg.reveal_forward_vgg(Rnet, _anc_img, anc_img, opt, True)
                    pos_img_rev, pos_lossR = steg.reveal_forward_vgg(Rnet, _pos_img, pos_img, opt, True)
                    neg_img_rev, neg_lossR = steg.reveal_forward_vgg(Rnet, _neg_img, neg_img, opt, True)

                    lossR = (anc_lossR + pos_lossR + neg_lossR) / 3

                    img_predA = nn.Softmax(dim=-1)(img_predA)
                    img_predB = nn.Softmax(dim=-1)(img_predB)
                    triplet_lossA = TripletLoss(opt.margin).forward(anc_embedA, pos_embedA, neg_embedA).cuda()
                    triplet_lossB = TripletLoss(opt.margin).forward(anc_embedB, pos_embedB, neg_embedB).cuda()
                    loss = torch.mean(torch.abs(img_predA - img_predB)) * 1000

                    lambda1 = ((lossH + lossR + content_loss) / (
                            loss + lossH + lossR + content_loss)).item()
                    lambda2 = ((loss + lossR + content_loss) / (
                            loss + lossH + lossR + content_loss)).item()
                    lambda3 = ((loss + lossH + content_loss) / (
                            loss + lossH + lossR + content_loss)).item()
                    lambda4 = ((loss + lossH + lossR) / (
                            loss + lossH + lossR + content_loss)).item()

                    if save_lossH.avg is None or save_lossH.avg == 0 or save_lossH.avg > 0.001:
                        steg_loss = 15 * lambda1 * (
                            loss) + 50 * (1 * lambda2 * lossH + 0.75 * lambda3 * lossR) + 0.1 * lambda4 * content_loss
                    else:
                        steg_loss = 15 * lambda1 * (loss) + (
                                1 * lambda2 * lossH + 1 * lambda3 * lossR + 0.1 * lambda4 * content_loss) * 1

                    cur_lossH.update(lossH.item())

                    if batch_idx % 8 == 0:
                        optimizerH.zero_grad()
                        optimizerR.zero_grad()
                        optimizerD1.zero_grad()
                        optimizerD2.zero_grad()
                    steg_loss.backward()
                    if batch_idx % 8 == 7:
                        optimizerH.step()
                        optimizerR.step()

                    II_lossH_sum += lossH.item()
                    II_lossR_sum += lossR.item()
                    II_loss_sum += loss.item()
                    II_content_loss_sum += content_loss.item()
                    II_total_loss_sum += steg_loss.item()
                    II_triplet_loss_sumA += triplet_lossA.item()
                    II_triplet_loss_sumB += triplet_lossB.item()

                    del anc_img, cover_anc_img, anc_img_rev
                    del pos_img, cover_pos_img, pos_img_rev
                    del neg_img, cover_neg_img, neg_img_rev
                    del features_x_anc, features_x_pos, features_x_neg
                    del features_y_anc, features_y_pos, features_y_neg
                    torch.cuda.empty_cache()

            if opt.is_valid == False:
                save_lossH = copy.deepcopy(cur_lossH)
                cur_lossH.reset()

                II_avg_lossH = II_lossH_sum / (batch_idx + 1)
                II_avg_lossR = II_lossR_sum / (batch_idx + 1)
                II_avg_adv_loss = II_loss_sum / (batch_idx + 1)
                II_avg_total_loss = II_total_loss_sum / (batch_idx + 1)
                II_avg_content_loss = II_content_loss_sum / (batch_idx + 1)
                II_avg_triplet_lossA = II_triplet_loss_sumA / (batch_idx + 1)
                II_avg_triplet_lossB = II_triplet_loss_sumB / (batch_idx + 1)

                logger.log_value('II ADV Loss', II_avg_adv_loss).step()
                logger.log_value('II lossH', II_avg_lossH).step()
                logger.log_value('II lossR', II_avg_lossR).step()
                logger.log_value('II content_loss', II_avg_content_loss).step()
                logger.log_value('II total_loss', II_avg_total_loss).step()

                labels = np.array([sublabel for label in labelsA for sublabel in label])
                distances = np.array([subdist for dist in distancesA for subdist in dist])

                tpr, fpr, accuracy1, val, val_std, far = evaluate(distances, labels)

                labels = np.array([sublabel for label in labelsB for sublabel in label])
                distances = np.array([subdist for dist in distancesB for subdist in dist])

                tpr, fpr, accuracy2, val, val_std, far = evaluate(distances, labels)

                print(80 * '=')
                print('Epoch [{}/{}]'.format(epoch, opt.num_epochs + opt.start_epoch - 1))
                # print("learingingH rate: {}".format(optimizerH.param_groups[0]['lr']))
                # print("learingingR rate: {}".format(optimizerR.param_groups[0]['lr']))
                # print("learingingD1 rate: {}".format(optimizerD1.param_groups[0]['lr']))
                # print("learingingD2 rate: {}".format(optimizerD2.param_groups[0]['lr']))

                print(' II  train set - triplet_lossA = {:.8f}'.format(np.mean(II_avg_triplet_lossA)))
                print(' II  train set - triplet_lossB = {:.8f}'.format(np.mean(II_avg_triplet_lossB)))
                print(' train set - AccuracyA = {:.8f}'.format(np.mean(accuracy1)))
                print(' train set - AccuracyB = {:.8f}'.format(np.mean(accuracy2)))
                print(' II  train set - LossH = {:.8f}'.format(II_avg_lossH))
                print(' II  train set - LossR = {:.8f}'.format(II_avg_lossR))
                print(' II  train set - content_loss = {:.8f}'.format(II_avg_content_loss))
                print(' II  train set - ADV Loss = {:.8f}'.format(II_avg_adv_loss))
                print(' II  train set - save_lossH.avg = {:.8f}'.format(save_lossH.avg))
                print(' ==> II  train set - total_Loss = {:.8f}'.format(II_avg_total_loss))

                with open(opt.logs_dir + '/train_classifier_log_epoch{}.txt'.format(epoch), 'w') as f:
                    f.write(str(epoch) + '\t' +
                            str(np.mean(accuracy1)) + '\t' +
                            str(np.mean(accuracy2)) + '\t' +
                            str(II_avg_total_loss))

        if epoch % 5 == 0:
            Hnet.eval()
            Rnet.eval()
            model.eval()
            modelA.eval()
            modelB.eval()
            noiser.eval()
            print(80 * '=')
            labelsA, distancesA = [], []
            labelsB, distancesB = [], []
            cross_entropy_loss_sumA = 0.0
            cross_entropy_loss_sumB = 0.0
            triplet_loss_sumA = 0.0
            triplet_loss_sumB = 0.0
            total_loss_sum = 0.0
            lossH_sum = 0.0
            lossR_sum = 0.0
            content_loss_sum = 0.0

            l2_dist = PairwiseDistance(2)

            for batch_idx, batch_sample in enumerate(data_loaders['valid']):
                if opt.is_test:
                    if opt.is_valid is False:
                        if batch_idx == 5:
                            break

                with torch.no_grad():
                    anc_img = batch_sample['anc_img'].cuda()
                    pos_img = batch_sample['pos_img'].cuda()
                    neg_img = batch_sample['neg_img'].cuda()

                    _anc_img = anc_img
                    _pos_img = pos_img
                    _neg_img = neg_img

                    pos_cls = batch_sample['pos_class'].cuda()
                    neg_cls = batch_sample['neg_class'].cuda()

                    features_x_anc, features_x_pos, features_x_neg = \
                        model(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                    anc_img, anc_lossH, _, cover_anc_img = steg.steg_forward_vggm(Hnet, train_cover_dataset,
                                                                                  anc_img, None,
                                                                                  None, opt, True)
                    pos_img, pos_lossH, _, cover_pos_img = steg.steg_forward_vggm(Hnet, train_cover_dataset,
                                                                                  pos_img, None,
                                                                                  None, opt, True)
                    neg_img, neg_lossH, _, cover_neg_img = steg.steg_forward_vggm(Hnet, train_cover_dataset,
                                                                                  neg_img, None,
                                                                                  None, opt, True)

                    anc_embedA, pos_embedA, neg_embedA = modelA(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(
                        3)
                    anc_embedB, pos_embedB, neg_embedB = modelB(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(
                        3)

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

                    features_y_anc, features_y_pos, features_y_neg = \
                        model(torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                    pos_img_predA, neg_img_predA = modelA.module.forward_classifier(
                        torch.cat([pos_embedA, neg_embedA], 0)).cuda().chunk(2)
                    pos_img_predB, neg_img_predB = modelB.module.forward_classifier(
                        torch.cat([pos_embedB, neg_embedB], 0)).cuda().chunk(2)

                    img_predA = torch.cat([pos_img_predA, neg_img_predA])
                    img_predB = torch.cat([pos_img_predB, neg_img_predB])
                    true_labels = torch.cat([Variable(pos_cls.cuda()), Variable(neg_cls.cuda())])
                    true_labels = torch.squeeze(true_labels, 1)

                    lossH = (anc_lossH + pos_lossH + neg_lossH) / 3

                    anc_img_rev, anc_lossR = steg.reveal_forward_vgg(Rnet, _anc_img, anc_img, opt, True)
                    pos_img_rev, pos_lossR = steg.reveal_forward_vgg(Rnet, _pos_img, pos_img, opt, True)
                    neg_img_rev, neg_lossR = steg.reveal_forward_vgg(Rnet, _neg_img, neg_img, opt, True)

                    lossR = (anc_lossR + pos_lossR + neg_lossR) / 3

                    criterion = nn.CrossEntropyLoss()
                    cross_entropy_lossA = criterion(img_predA.cuda(), true_labels.cuda())
                    cross_entropy_lossB = criterion(img_predB.cuda(), true_labels.cuda())
                    triplet_lossA = TripletLoss(opt.margin).forward(anc_embedA, pos_embedA, neg_embedA).cuda()
                    triplet_lossB = TripletLoss(opt.margin).forward(anc_embedB, pos_embedB, neg_embedB).cuda()

                    content_loss = (mse_loss(features_x_anc, features_y_anc) \
                                    + mse_loss(features_x_pos, features_y_pos) \
                                    + mse_loss(features_x_neg, features_y_neg)) / 3

                    lossA = cross_entropy_lossA + 2 * triplet_lossA
                    lossB = cross_entropy_lossB + 2 * triplet_lossB
                    total_loss = lossA + lossB + lossH + lossR

                    cross_entropy_loss_sumA += cross_entropy_lossA.item()
                    cross_entropy_loss_sumB += cross_entropy_lossB.item()
                    triplet_loss_sumA += triplet_lossA.item()
                    triplet_loss_sumB += triplet_lossB.item()
                    lossH_sum += lossH.item()
                    lossR_sum += lossR.item()
                    total_loss_sum += total_loss.item()
                    content_loss_sum += content_loss.item()

                    if opt.is_valid == False:
                        if batch_idx == 0:
                            _anc_img = steg.tensor_image_resize(_anc_img.cpu()).cuda()
                            steg.save_image(_anc_img[0], '{}/{}_img.png'.format(opt.output_dir, epoch))
                            steg.save_image(anc_img[0],
                                            '{}/{}_container_img.png'.format(opt.output_dir, epoch))
                            steg.save_image(anc_img_rev[0],
                                            '{}/{}_reaveal_img.png'.format(opt.output_dir, epoch))
                            print('image saved: {}/{}_img.png'.format(opt.output_dir, epoch))
                            print('image saved: {}/{}_container_img.png'.format(opt.output_dir, epoch))
                            print('image saved: {}/{}_reveal_img.png'.format(opt.output_dir, epoch))

                    del anc_img, pos_img, neg_img
                    del _anc_img, _pos_img, _neg_img
                    del cover_anc_img, cover_pos_img, cover_neg_img
                    del anc_img_rev, pos_img_rev, neg_img_rev
                    torch.cuda.empty_cache()

            avg_cross_entropy_lossA = cross_entropy_loss_sumA / (batch_idx + 1)
            avg_triplet_lossA = triplet_loss_sumA / (batch_idx + 1)
            avg_cross_entropy_lossB = cross_entropy_loss_sumB / (batch_idx + 1)
            avg_triplet_lossB = triplet_loss_sumB / (batch_idx + 1)
            avg_lossH = lossH_sum / (batch_idx + 1)
            avg_lossR = lossR_sum / (batch_idx + 1)
            avg_content_loss = content_loss_sum / (batch_idx + 1)
            avg_total_loss = total_loss_sum / (batch_idx + 1)

            labels = np.array([sublabel for label in labelsA for sublabel in label])
            distances = np.array([subdist for dist in distancesA for subdist in dist])

            tpr, fpr, accuracy1, val, val_std, far = evaluate(distances, labels)

            labels = np.array([sublabel for label in labelsB for sublabel in label])
            distances = np.array([subdist for dist in distancesB for subdist in dist])

            tpr, fpr, accuracy2, val, val_std, far = evaluate(distances, labels)

            print(' valid set - Cross Entropy LossA = {:.8f}'.format(avg_cross_entropy_lossA))
            print(' valid set - Cross Entropy LossB = {:.8f}'.format(avg_cross_entropy_lossB))
            print(' valid set - Triplet LossA = {:.8f}'.format(avg_triplet_lossA))
            print(' valid set - Triplet LossB = {:.8f}'.format(avg_triplet_lossB))
            print(' valid set - AccuracyA = {:.8f}'.format(np.mean(accuracy1)))
            print(' valid set - AccuracyB = {:.8f}'.format(np.mean(accuracy2)))
            print(' ==> valid set - LossH = {:.8f}'.format(avg_lossH))
            print(' ==> valid set - LossR = {:.8f}'.format(avg_lossR))
            print(' ==> valid set - content loss = {:.8f}'.format(avg_content_loss))
            print(' ==> valid set - Total Loss = {:.8f}'.format(avg_total_loss))

            logger.log_value('valid_cross_entropy_lossA', avg_cross_entropy_lossA).step()
            logger.log_value('valid_triplet_lossA', avg_triplet_lossA).step()
            logger.log_value('valid_accuracy1', np.mean(accuracy1)).step()
            logger.log_value('valid_cross_entropy_lossB', avg_cross_entropy_lossB).step()
            logger.log_value('valid_triplet_lossB', avg_triplet_lossB).step()
            logger.log_value('valid_accuracy2', np.mean(accuracy2)).step()
            logger.log_value('valid_lossH', avg_lossH).step()
            logger.log_value('valid_lossR', avg_lossR).step()
            logger.log_value('valid_total_loss', avg_total_loss).step()

        if opt.is_valid == False:
            if epoch % 5 == 0:
                print("save at", opt.ckpt_dir + '/checkpoint_epoch{}.pth'.format(epoch))
                torch.save({'epoch': epoch,
                            'Hnet_state_dict': Hnet.state_dict(),
                            'Rnet_state_dict': Rnet.state_dict(),
                            'clsA_state_dict': modelA.state_dict(),
                            'clsB_state_dict': modelB.state_dict()},
                           opt.ckpt_dir + '/checkpoint_epoch{}.pth'.format(epoch))


def test(**kwargs):
    opt.load_steg_model_dir = './model_zoo/steg_full_high_acc_model.pth'
    opt.load_classifier_model_dir1 = './model_zoo/steg_full_high_acc_model.pth'
    opt.load_classifier_model_dir2 = './model_zoo/steg_full_high_acc_model.pth'

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
    Rnet = oriRevealNet()
    Rnet = Rnet.cuda()

    cover_dataset = MyImageFolder(
        opt.cover_dir,
        transforms.Compose([
            transforms.Resize([opt.image_size, opt.image_size]),
            transforms.ToTensor(),
        ]))

    if opt.load_steg_model_dir != '':
        checkpoint = torch.load(opt.load_steg_model_dir, map_location={'cuda:1': opt.device})
        Hnet.load_state_dict(preprocess_ckpt(checkpoint['Hnet_state_dict']))
        Rnet.load_state_dict(preprocess_ckpt(checkpoint['Rnet_state_dict']))

    if opt.load_classifier_model_dir1 != '':
        checkpoint = torch.load(opt.load_classifier_model_dir1, map_location={'cuda:1': opt.device})
        modelA.load_state_dict(preprocess_ckpt(checkpoint['clsA_state_dict']))

    if opt.load_classifier_model_dir2 != '':
        checkpoint = torch.load(opt.load_classifier_model_dir2, map_location={'cuda:1': opt.device})
        modelB.load_state_dict(preprocess_ckpt(checkpoint['clsB_state_dict']))

    mse_loss = torch.nn.MSELoss()

    data_loaders, data_size = get_dataloader(opt.train_root_dir, opt.valid_root_dir,
                                             opt.train_csv_name, opt.valid_csv_name,
                                             opt.num_train_triplets, opt.num_valid_triplets,
                                             opt.batch_size, opt.num_workers, False)

    Hnet.eval()
    Rnet.eval()
    modelA.eval()
    modelB.eval()
    noiser.eval()
    labelsA, distancesA = [], []
    labelsB, distancesB = [], []
    vis_mse_loss_sum = []
    rec_mse_loss_sum = []

    l2_dist = PairwiseDistance(2)

    for batch_idx, batch_sample in enumerate(data_loaders['valid']):
        with torch.no_grad():
            anc_img = batch_sample['anc_img'].cuda()
            pos_img = batch_sample['pos_img'].cuda()
            neg_img = batch_sample['neg_img'].cuda()

            _anc_img = anc_img
            _pos_img = pos_img
            _neg_img = neg_img

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

            anc_img_rev, anc_lossR = steg.reveal_forward_vgg(Rnet, _anc_img, anc_img, opt, True)
            pos_img_rev, pos_lossR = steg.reveal_forward_vgg(Rnet, _pos_img, pos_img, opt, True)
            neg_img_rev, neg_lossR = steg.reveal_forward_vgg(Rnet, _neg_img, neg_img, opt, True)

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

            vis_img_mse_loss = (mse_loss(anc_img, cover_anc_img) + mse_loss(pos_img, cover_pos_img) + mse_loss(neg_img,
                                                                                                               cover_neg_img)) / 3
            rec_img_mse_loss = (mse_loss(_anc_img, anc_img_rev) + mse_loss(_pos_img, pos_img_rev) + mse_loss(_neg_img,
                                                                                                             neg_img_rev)) / 3

            vis_mse_loss_sum.append(vis_img_mse_loss.item())
            rec_mse_loss_sum.append(rec_img_mse_loss.item())

    avg_vis_mse_loss = np.mean(vis_mse_loss_sum)
    vis_psnr = -10 * np.log10(avg_vis_mse_loss)

    avg_rec_mse_loss = np.mean(rec_mse_loss_sum)
    rec_psnr = -10 * np.log10(avg_rec_mse_loss)

    labels = np.array([sublabel for label in labelsA for sublabel in label])
    distances = np.array([subdist for dist in distancesA for subdist in dist])

    tpr, fpr, accuracy1, val, val_std, far = evaluate(distances, labels)

    labels = np.array([sublabel for label in labelsB for sublabel in label])
    distances = np.array([subdist for dist in distancesB for subdist in dist])

    tpr, fpr, accuracy2, val, val_std, far = evaluate(distances, labels)

    print('  valid set - Accuracy1 = {:.8f}'.format(np.mean(accuracy1)))
    print('  valid set - Accuracy2 = {:.8f}'.format(np.mean(accuracy2)))
    print(' ==>  valid set - PSNR (cover and container) = {:.3f}'.format(vis_psnr))
    print(' ==>  valid set - PSNR (secret and restore) = {:.3f}'.format(rec_psnr))


def visualize(output_path='outputs', **kwargs):
    try:
        os.mkdir(output_path)
    except Exception:
        pass

    opt.output_dir = output_path

    opt.load_steg_model_dir = './model_zoo/steg_full_high_psnr_model.pth'

    opt.parse(kwargs)
    opt.device = 'cuda:0'
    print(opt.device)

    NoiseArgParser()(opt.noise, opt)

    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Hnet = Hnet.cuda()
    Rnet = oriRevealNet()
    Rnet = Rnet.cuda()
    noiser = Noiser(opt.noise, opt.device)
    noiser = noiser.cuda()

    cover_dataset = MyImageFolder(
        opt.cover_dir,
        transforms.Compose([
            transforms.Resize([opt.image_size, opt.image_size]),
            transforms.ToTensor(),
        ]))

    if opt.load_steg_model_dir != '':
        checkpoint = torch.load(opt.load_steg_model_dir, map_location={'cuda:1': opt.device})
        Hnet.load_state_dict(preprocess_ckpt(checkpoint['Hnet_state_dict']))
        Rnet.load_state_dict(preprocess_ckpt(checkpoint['Rnet_state_dict']))

    Hnet = DataParallel(Hnet)
    Rnet = DataParallel(Rnet)
    noiser = DataParallel(noiser)

    data_loaders, data_size = get_dataloader(opt.train_root_dir, opt.valid_root_dir, opt.train_csv_name,
                                             opt.valid_csv_name, opt.num_train_triplets, opt.num_valid_triplets,
                                             opt.batch_size, opt.num_workers, False)

    Hnet.eval()
    Rnet.eval()
    noiser.eval()

    for batch_idx, batch_sample in enumerate(data_loaders['valid']):
        with torch.no_grad():
            anc_img = batch_sample['anc_img'].cuda()

            anc_img_con, anc_lossH, _, anc_cover = steg.steg_forward_vggm(Hnet, cover_dataset, anc_img,
                                                                          None, None, opt, True)
            anc_img_rev, anc_lossR = steg.reveal_forward_vgg(Rnet, anc_img, anc_img_con, opt, True)

            steg.save_image(anc_img, opt.output_dir + '/secret_%d.png' % batch_idx)
            steg.save_image(anc_cover, opt.output_dir + '/clean_%d.png' % batch_idx)
            steg.save_image(anc_img_con, opt.output_dir + '/stego_%d.png' % batch_idx)
            steg.save_image(anc_img_rev, opt.output_dir + '/reveal_%d.png' % batch_idx)

            break


if __name__ == '__main__':
    import fire
    
    fire.Fire()
