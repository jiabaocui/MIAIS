import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.modules.distance import PairwiseDistance
from torchvision import transforms

sys.path.append(os.path.abspath('./steg'))
import steg
from models import UnetGenerator
from models import oriRevealNet
from models import FaceNetInceptionModel
from data import MyImageFolder
from data import get_dataloader
from utils import Logger
from utils import evaluate
from utils import TripletLoss
from utils.AFD_config import opt


def train(**kwargs):
    opt.parse(kwargs)
    opt.num_classes = 1662
    opt.start_epoch = 0
    opt.batch_size = 2
    opt.steg_lr = 1e-3
    opt.cover_dir = '/4T/liangli/ste/data/cover'

    opt.load_model_dir = '/4T/liangli/ste/data/checkpoint_epoch4230.pth'
    opt.load_model2_dir = '/4T/liangli/ste/data/checkpoint_epoch4230.pth'
    opt.load_Hnet_dir = ''
    opt.load_Rnet_dir = ''

    opt.num_train_triplets = 2000
    opt.num_valid_triplets = 2000

    opt.root_dir = '/4T/liangli/ste/output/MIAIS_jointly3/'
    opt.ckpt_dir = opt.root_dir + 'ckpt'
    opt.logs_dir = opt.root_dir + 'logs'
    opt.output_dir = opt.root_dir + 'output'
    opt.device = 'cuda:3'

    opt.train_root_dir = '/4T/liangli/ste/data/AFD_part1'
    opt.valid_root_dir = '/4T/liangli/ste/data/AFD_part2'
    opt.train_csv_name = './data/AFD_part1.csv'
    opt.valid_csv_name = './data/AFD_part2.csv'

    opt.margin = 1.2

    opt.is_test = False

    opt.print_config()
    print(opt.device)

    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Hnet.to(opt.device)
    Rnet = oriRevealNet()
    Rnet.to(opt.device)

    model = FaceNetInceptionModel(opt.embedding_size, opt.num_classes, pretrained=False)
    model.to(opt.device)

    model2 = FaceNetInceptionModel(opt.embedding_size, opt.num_classes, pretrained=False)
    model2.to(opt.device)

    if opt.load_Hnet_dir != '':
        checkpoint = torch.load(opt.load_Hnet_dir, map_location={'cuda:0': opt.device})
        print('*******')
        print('load model from' + opt.load_Hnet_dir)
        print('*******')
        Hnet.load_state_dict(checkpoint['Hnet_state_dict'])
        # Hnet.load_state_dict(checkpoint)

    if opt.load_Rnet_dir != '':
        checkpoint = torch.load(opt.load_Rnet_dir)
        print('*******')
        print('load model from' + opt.load_Rnet_dir)
        print('*******')
        Rnet.load_state_dict(checkpoint['Rnet_state_dict'])

    if opt.load_model_dir != '':
        checkpoint = torch.load(opt.load_model_dir, map_location={'cuda:0': opt.device})
        print('*******')
        print('load model from' + opt.load_model_dir)
        print('*******')
        model.load_state_dict(checkpoint['state_dict'])

    if opt.load_model2_dir != '':
        checkpoint = torch.load(opt.load_model2_dir, map_location={'cuda:0': opt.device})
        print('*******')
        print('load model from' + opt.load_model2_dir)
        print('*******')
        model2.load_state_dict(checkpoint['state_dict'])

    model2.eval()
    # del checkpoint
    # torch.cuda.empty_cache()

    cover_dataset = MyImageFolder(
        opt.cover_dir,
        transforms.Compose([
            transforms.Resize([opt.image_size, opt.image_size]),
            transforms.ToTensor(),
        ]))

    optimizerH = optim.Adam(Hnet.parameters(), lr=opt.steg_lr * 0.5, betas=(0.5, 0.999))
    optimizerR = optim.Adam(Rnet.parameters(), lr=opt.steg_lr * 0.5, betas=(0.5, 0.999))
    optimizer = optim.Adam(model.parameters(), lr=opt.steg_lr)

    logger = Logger(opt.logs_dir)

    mse_loss = torch.nn.MSELoss()
    l2_dist = PairwiseDistance(2)

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):
        Hnet.train()
        Rnet.train()
        model.train()
        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch, opt.num_epochs + opt.start_epoch - 1))

        lossH_sum = 0.0
        lossR_sum = 0.0
        content_loss_sum = 0.0

        labels, distances = [], []
        cross_entropy_loss_sum = 0.0
        triplet_loss_sum = 0.0
        data_loaders, data_size = get_dataloader(opt.train_root_dir, opt.valid_root_dir,
                                                 opt.train_csv_name, opt.valid_csv_name,
                                                 opt.num_train_triplets, opt.num_valid_triplets,
                                                 opt.batch_size, opt.num_workers, False)
        for batch_idx, batch_sample in enumerate(data_loaders['train']):
            # break
            # if batch_idx == 100:
            #     break
            if opt.is_test is True:
                if batch_idx > 5:
                    break

            with torch.set_grad_enabled(True):
                anc_img = batch_sample['anc_img'].to(opt.device)
                pos_img = batch_sample['pos_img'].to(opt.device)
                neg_img = batch_sample['neg_img'].to(opt.device)

                features_x_anc, features_x_pos, features_x_neg = model2(
                    torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                pos_cls = batch_sample['pos_class'].to(opt.device)
                neg_cls = batch_sample['neg_class'].to(opt.device)

                anc_img_con, anc_lossH, _, anc_cover = steg.steg_forward_vggm(Hnet, cover_dataset, anc_img,
                                                                              None, None, opt, True)
                pos_img_con, pos_lossH, _, pos_cover = steg.steg_forward_vggm(Hnet, cover_dataset, pos_img,
                                                                              None, None, opt, True)
                neg_img_con, neg_lossH, _, neg_cover = steg.steg_forward_vggm(Hnet, cover_dataset, neg_img,
                                                                              None, None, opt, True)

                features_y_anc, features_y_pos, features_y_neg = model2(
                    torch.cat([anc_img_con, pos_img_con, neg_img_con], 0)).chunk(3)

                content_loss = (mse_loss(features_x_anc, features_y_anc) + mse_loss(features_x_pos,
                                                                                    features_y_pos) + mse_loss(
                    features_x_neg, features_y_neg))

                lossH = (anc_lossH + pos_lossH + neg_lossH) / 3

                anc_embed, pos_embed, neg_embed = model(torch.cat([anc_img_con, pos_img_con, neg_img_con], 0)).chunk(3)

                anc_img_rev, anc_lossR = steg.reveal_forward_vgg(Rnet, anc_img, anc_img_con, opt, True)
                pos_img_rev, pos_lossR = steg.reveal_forward_vgg(Rnet, pos_img, pos_img_con, opt, True)
                neg_img_rev, neg_lossR = steg.reveal_forward_vgg(Rnet, neg_img, neg_img_con, opt, True)

                lossR = (anc_lossR + pos_lossR + neg_lossR) / 3

                anc_img_pred, pos_img_pred, neg_img_pred = model.forward_classifier(
                    torch.cat([anc_embed, pos_embed, neg_embed], 0)).to(opt.device).chunk(3)

                img_pred = torch.cat([anc_img_pred, pos_img_pred, neg_img_pred])
                true_labels = torch.cat([Variable(pos_cls), Variable(pos_cls), Variable(neg_cls)])
                true_labels = torch.squeeze(true_labels, 1)

                criterion = nn.CrossEntropyLoss()
                cross_entropy_loss = criterion(img_pred.to(opt.device), true_labels.to(opt.device))
                triplet_loss = TripletLoss(opt.margin).forward(anc_embed, pos_embed, neg_embed).to(opt.device)

                loss = cross_entropy_loss + 2 * triplet_loss + 3 * (lossH + 0.75 * lossR) + 0.1 * content_loss

                optimizerH.zero_grad()
                optimizerR.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizerH.step()
                optimizerR.step()
                optimizer.step()

                dists = l2_dist.forward(anc_embed, pos_embed)
                distances.append(dists.data.cpu().numpy())
                labels.append(np.ones(dists.size(0)))

                dists = l2_dist.forward(anc_embed, neg_embed)
                distances.append(dists.data.cpu().numpy())
                labels.append(np.zeros(dists.size(0)))

                cross_entropy_loss_sum += cross_entropy_loss.item()
                triplet_loss_sum += triplet_loss.item()
                lossH_sum += lossH.item()
                lossR_sum += lossR.item()
                content_loss_sum += content_loss.item()

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)

        avg_cross_entropy_loss = cross_entropy_loss_sum / (batch_idx + 1)
        avg_triplet_loss = triplet_loss_sum / (batch_idx + 1)
        avg_lossH = lossH_sum / (batch_idx + 1)
        avg_lossR = lossR_sum / (batch_idx + 1)
        avg_content_loss = content_loss_sum / (batch_idx + 1)

        print('train set - Cross Entropy Loss = {:.8f}'.format(avg_cross_entropy_loss))
        print('train set - Triplet Loss = {:.8f}'.format(avg_triplet_loss))
        print('train set - Accuracy = {:.8f}'.format(np.mean(accuracy)))
        print('train set - lossH = {:.8f}'.format(avg_lossH))
        print('train set - lossR = {:.8f}'.format(avg_lossR))
        print('train set - content_loss = {:.8f}'.format(avg_content_loss))

        logger.log_value('train_cross_entropy_loss', avg_cross_entropy_loss).step()
        logger.log_value('train_triplet_loss', avg_triplet_loss).step()
        logger.log_value('train_accuracy', np.mean(accuracy)).step()
        logger.log_value('train_avg_lossH', avg_lossH).step()
        logger.log_value('train_avg_lossR', avg_lossR).step()
        logger.log_value('train_avg_content_loss', avg_content_loss).step()

        if epoch % 5 == 0:
            Hnet.eval()
            Rnet.eval()
            model.eval()

            labels, distances = [], []
            cross_entropy_loss_sum = 0.0
            triplet_loss_sum = 0.0
            lossH_sum = 0.0
            lossR_sum = 0.0
            content_loss_sum = 0.0

            for batch_idx, batch_sample in enumerate(data_loaders['valid']):
                # if batch_idx == 100:
                #     break
                if opt.is_test is True:
                    if batch_idx > 5:
                        break

                anc_img = batch_sample['anc_img'].to(opt.device)
                pos_img = batch_sample['pos_img'].to(opt.device)
                neg_img = batch_sample['neg_img'].to(opt.device)
                with torch.set_grad_enabled(True):

                    anc_img_con, anc_lossH, cover_anc_idx, anc_cover = steg.steg_forward_vggm(Hnet, cover_dataset,
                                                                                              anc_img,
                                                                                              None, None, opt, True)
                    pos_img_con, pos_lossH, cover_pos_idx, pos_cover = steg.steg_forward_vggm(Hnet, cover_dataset,
                                                                                              pos_img,
                                                                                              None, None, opt, True)
                    neg_img_con, neg_lossH, cover_neg_idx, neg_cover = steg.steg_forward_vggm(Hnet, cover_dataset,
                                                                                              neg_img,
                                                                                              None, None, opt, True)
                    features_x_anc, features_x_pos, features_x_neg = model2(
                        torch.cat([anc_img, pos_img, neg_img], 0)).chunk(3)

                    features_y_anc, features_y_pos, features_y_neg = model2(
                        torch.cat([anc_img_con, pos_img_con, neg_img_con], 0)).chunk(3)

                    content_loss = (mse_loss(features_x_anc, features_y_anc) + mse_loss(features_x_pos,
                                                                                        features_y_pos) + mse_loss(
                        features_x_neg, features_y_neg))

                    lossH = (anc_lossH + pos_lossH + neg_lossH) / 3

                    anc_embed, pos_embed, neg_embed = model(
                        torch.cat([anc_img_con, pos_img_con, neg_img_con], 0)).chunk(3)

                    pos_cls = batch_sample['pos_class'].to(opt.device)
                    neg_cls = batch_sample['neg_class'].to(opt.device)

                    anc_img_pred, pos_img_pred, neg_img_pred = model.forward_classifier(
                        torch.cat([anc_embed, pos_embed, neg_embed], 0)).to(opt.device).chunk(3)

                    anc_img_rev, anc_lossR = steg.reveal_forward_vgg(Rnet, anc_img, anc_img_con, opt, True)
                    pos_img_rev, pos_lossR = steg.reveal_forward_vgg(Rnet, pos_img, pos_img_con, opt, True)
                    neg_img_rev, neg_lossR = steg.reveal_forward_vgg(Rnet, neg_img, neg_img_con, opt, True)

                    lossR = (anc_lossR + pos_lossR + neg_lossR) / 3

                    img_pred = torch.cat([anc_img_pred, pos_img_pred, neg_img_pred])
                    true_labels = torch.cat([Variable(pos_cls), Variable(pos_cls), Variable(neg_cls)])
                    true_labels = torch.squeeze(true_labels, 1)

                    criterion = nn.CrossEntropyLoss()
                    cross_entropy_loss = criterion(img_pred.to(opt.device), true_labels.to(opt.device))
                    triplet_loss = TripletLoss(opt.margin).forward(anc_embed, pos_embed, neg_embed).to(opt.device)

                    dists = l2_dist.forward(anc_embed, pos_embed)
                    distances.append(dists.data.cpu().numpy())
                    labels.append(np.ones(dists.size(0)))

                    dists = l2_dist.forward(anc_embed, neg_embed)
                    distances.append(dists.data.cpu().numpy())
                    labels.append(np.zeros(dists.size(0)))

                    cross_entropy_loss_sum += cross_entropy_loss.item()
                    triplet_loss_sum += triplet_loss.item()
                    lossH_sum += lossH.item()
                    lossR_sum += lossR.item()
                    content_loss_sum += content_loss.item()

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])

            tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)

            avg_cross_entropy_loss = cross_entropy_loss_sum / (batch_idx + 1)
            avg_triplet_loss = triplet_loss_sum / (batch_idx + 1)
            avg_lossH = lossH_sum / (batch_idx + 1)
            avg_lossR = lossR_sum / (batch_idx + 1)
            avg_content_loss = content_loss_sum / (batch_idx + 1)

            print('valid set - Cross Entropy Loss = {:.8f}'.format(avg_cross_entropy_loss))
            print('valid set - Triplet Loss = {:.8f}'.format(avg_triplet_loss))
            print('valid set - Accuracy = {:.8f}'.format(np.mean(accuracy)))
            print('valid set - lossH = {:.8f}'.format(avg_lossH))
            print('valid set - lossR = {:.8f}'.format(avg_lossR))
            print('valid set - content_loss = {:.8f}'.format(avg_content_loss))

            logger.log_value('valid_cross_entropy_loss', avg_cross_entropy_loss).step()
            logger.log_value('valid_triplet_loss', avg_triplet_loss).step()
            logger.log_value('valid_accuracy', np.mean(accuracy)).step()
            logger.log_value('valid_avg_lossH', avg_lossH).step()
            logger.log_value('valid_avg_lossR', avg_lossR).step()
            logger.log_value('valid_avg_content_loss', avg_content_loss).step()

            print("save at", opt.ckpt_dir + '/checkpoint_epoch{}.pth'.format(epoch))
            torch.save({'epoch': epoch,
                        'Hnet_state_dict': Hnet.state_dict(),
                        'Rnet_state_dict': Rnet.state_dict(),
                        'cls_state_dict': model.state_dict()},
                       opt.ckpt_dir + '/checkpoint_epoch{}.pth'.format(epoch))


if __name__ == '__main__':
    import fire

    fire.Fire()
