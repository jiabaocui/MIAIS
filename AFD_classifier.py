import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.distance import PairwiseDistance
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

sys.path.append(os.path.abspath('./steg'))
from models import FaceNetInceptionModel
from data import get_dataloader
from utils import Logger
from utils.AFD_config import opt
from utils import evaluate
from utils import TripletLoss


def train(**kwargs):
    opt.parse(kwargs)

    opt.batch_size = 48
    opt.steg_lr = 1e-4

    opt.num_train_triplets = 5000
    opt.num_valid_triplets = 2000

    opt.root_dir = './results/'
    opt.ckpt_dir = opt.root_dir + 'ckpt'
    opt.logs_dir = opt.root_dir + 'logs'
    opt.output_dir = opt.root_dir + 'output'
    opt.device = 'cuda:0'

    opt.is_test = False
    opt.is_valid = False

    # ***************************************************************************************** #

    opt.print_config()
    print(opt.device)

    model = FaceNetInceptionModel(opt.embedding_size, opt.num_classes, pretrained=False)
    model.to(opt.device)

    if opt.load_model_dir != '':
        checkpoint = torch.load(opt.load_model_dir, map_location={'cuda:1': opt.device})
        print('*******')
        print('load model from' + opt.load_model_dir)
        print('*******')
        model.load_state_dict(checkpoint['cls_state_dict'])

    optimizer = optim.Adam(model.parameters(), lr=opt.steg_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30, verbose=True)

    if opt.start_epoch == 0:
        logger = Logger(opt.logs_dir, True)
    else:
        logger = Logger(opt.logs_dir)

    l2_dist = PairwiseDistance(2)

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):
        model.train()
        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch, opt.num_epochs + opt.start_epoch - 1))

        labels, distances = [], []
        cross_entropy_loss_sum = 0.0
        triplet_loss_sum = 0.0

        data_loaders, data_size = get_dataloader(opt.train_root_dir, opt.valid_root_dir,
                                                 opt.train_csv_name, opt.valid_csv_name,
                                                 opt.num_train_triplets, opt.num_valid_triplets,
                                                 opt.batch_size, opt.num_workers)

        for batch_idx, batch_sample in enumerate(data_loaders['train']):
            if opt.is_test is True:
                if batch_idx > 5:
                    break
            if opt.is_valid is True:
                break

            with torch.set_grad_enabled(True):
                anc_img = batch_sample['anc_img'].to(opt.device)
                pos_img = batch_sample['pos_img'].to(opt.device)
                neg_img = batch_sample['neg_img'].to(opt.device)

                anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                pos_cls = batch_sample['pos_class'].to(opt.device)
                neg_cls = batch_sample['neg_class'].to(opt.device)

                anc_img_pred = model.forward_classifier(anc_embed).to(opt.device)
                pos_img_pred = model.forward_classifier(pos_embed).to(opt.device)
                neg_img_pred = model.forward_classifier(neg_embed).to(opt.device)

                img_pred = torch.cat([anc_img_pred, pos_img_pred, neg_img_pred])
                true_labels = torch.cat([Variable(pos_cls), Variable(pos_cls), Variable(neg_cls)])
                true_labels = torch.squeeze(true_labels, 1)

                criterion = nn.CrossEntropyLoss()
                cross_entropy_loss = criterion(img_pred.to(opt.device), true_labels.to(opt.device))
                triplet_loss = TripletLoss(opt.margin).forward(anc_embed, pos_embed, neg_embed).to(opt.device)

                steg_loss = 0.01 * cross_entropy_loss + 2 * triplet_loss

                optimizer.zero_grad()
                steg_loss.backward()
                optimizer.step()

                dists = l2_dist.forward(anc_embed, pos_embed)
                distances.append(dists.data.cpu().numpy())
                labels.append(np.ones(dists.size(0)))

                dists = l2_dist.forward(anc_embed, neg_embed)
                distances.append(dists.data.cpu().numpy())
                labels.append(np.zeros(dists.size(0)))

                cross_entropy_loss_sum += cross_entropy_loss.item()
                triplet_loss_sum += triplet_loss.item()

        if opt.is_valid == False:
            avg_cross_entropy_loss = cross_entropy_loss_sum / math.ceil(batch_idx + 1)
            avg_triplet_loss = triplet_loss_sum / math.ceil(batch_idx + 1)
            avg_cross_entropy_loss /= 3
            avg_triplet_loss /= 3

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])

            tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)

            print('train set - Cross Entropy Loss = {:.8f}'.format(avg_cross_entropy_loss))
            print('train set - Triplet Loss = {:.8f}'.format(avg_triplet_loss))
            print('train set - Accuracy = {:.8f}'.format(np.mean(accuracy)))

            logger.log_value('train_cross_entropy_loss', avg_cross_entropy_loss).step()
            logger.log_value('train_triplet_loss', avg_triplet_loss).step()
            logger.log_value('train_accuracy', np.mean(accuracy)).step()

        if epoch % 2 == 0:
            model.eval()

            labels, distances = [], []
            cross_entropy_loss_sum = 0.0
            triplet_loss_sum = 0.0

            for batch_idx, batch_sample in enumerate(data_loaders['valid']):
                if opt.is_test is True:
                    if opt.is_valid is False:
                        if batch_idx > 5:
                            break

                with torch.set_grad_enabled(True):
                    anc_img = batch_sample['anc_img'].to(opt.device)
                    pos_img = batch_sample['pos_img'].to(opt.device)
                    neg_img = batch_sample['neg_img'].to(opt.device)

                    anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                    pos_cls = batch_sample['pos_class'].to(opt.device)
                    neg_cls = batch_sample['neg_class'].to(opt.device)

                    anc_img_pred = model.forward_classifier(anc_embed).to(opt.device)
                    pos_img_pred = model.forward_classifier(pos_embed).to(opt.device)
                    neg_img_pred = model.forward_classifier(neg_embed).to(opt.device)

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

            avg_cross_entropy_loss = cross_entropy_loss_sum / math.ceil(batch_idx + 1)
            avg_triplet_loss = triplet_loss_sum / math.ceil(batch_idx + 1)
            avg_cross_entropy_loss /= 3
            avg_triplet_loss /= 3

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])

            tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)

            print('valid set - Cross Entropy Loss = {:.8f}'.format(avg_cross_entropy_loss))
            print('valid set - Triplet Loss = {:.8f}'.format(avg_triplet_loss))
            print('valid set - Accuracy = {:.8f}'.format(np.mean(accuracy)))

            logger.log_value('valid_cross_entropy_loss', avg_cross_entropy_loss).step()
            logger.log_value('valid_triplet_loss', avg_triplet_loss).step()
            logger.log_value('valid_accuracy', np.mean(accuracy)).step()

            scheduler.step(avg_triplet_loss)

        if opt.is_valid == False:
            if epoch % 5 == 0:
                torch.save({'epoch': epoch,
                            'cls_state_dict': model.state_dict()},
                           opt.ckpt_dir + '/cls_checkpoint{}.pth'.format(epoch))


if __name__ == '__main__':
    import fire

    fire.Fire()
