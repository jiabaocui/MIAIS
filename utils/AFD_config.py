import warnings
import torch


class DefaultConfig(object):

    # Change to your own path
    cover_dir = '/mnt/data2/pengyi/encryption/cover'
    train_root_dir = '/mnt/data2/pengyi/encryption/AFD_part1'
    valid_root_dir = '/mnt/data2/pengyi/encryption/AFD_part2'

    train_csv_name = './data/AFD_part1.csv'
    valid_csv_name = './data/AFD_part2.csv'

    model = 'FaceNetInceptionModel'
    start_epoch = 0
    num_epochs = 150

    num_classes = 1662
    num_train_triplets = 2000
    num_valid_triplets = 1000

    embedding_size = 128
    batch_size = 16
    num_workers = 8
    # learning_rate = 0.001
    margin = 1.2

    image_size = 256

    decay_betas = 0.5
    beta = 0.75
    # steg_lr = 1.25e-04 / 2
    steg_lr = 1e-3

    theta = 1
    gamma = 1

    load_model_dir = ''
    load_Hnet_dir = ''
    load_Rnet_dir = ''
    load_optimizerR_dir = ''
    load_schedulerR_dir = ''


    root_dir = ''
    ckpt_dir = root_dir + 'ckpt'
    logs_dir = root_dir + 'logs'
    output_dir = root_dir + 'output'

    is_Hnet_act = True
    is_Rnet_act = True

    use_gpu = True
    ngpu = 1

    stage = 1

    is_test = False
    is_valid = False

    device = torch.device('cuda:0')

    noise = 'cropout((0.85, 1.),(0.85, 1.))+dropout(0.7, 1.)+jpeg()'

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        if not opt.use_gpu:
            opt.device = torch.device('cpu')

    def print_config(self):
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
opt.margin = 1.2

opt.num_train_triplets = 2000
opt.num_valid_triplets = 2000
opt.batch_size = 16
opt.start_epoch = 0

opt.learning_rate = 1e-3

# Change to your own path
opt.valid_cover_dir = '/mnt/data2/pengyi/encryption/cover'
opt.train_cover_dir = '/mnt/data2/pengyi/encryption/cover'

opt.load_steg_model_dir = ''
opt.load_classifier_model_dir1 = ''
opt.load_classifier_model_dir2 = ''
opt.load_classifier_model_dir = ''
opt.load_dir = ''

opt.root_dir = ''
opt.ckpt_dir = opt.root_dir + 'ckpt'
opt.logs_dir = opt.root_dir + 'logs'
opt.output_dir = opt.root_dir + 'output'


opt.device = ""