import torch


class ConfigKos:
    dataset = 'Ru_KOS'
    num_chars = 166
    data_folder_path = './RIMES/'  # relative to ./data/
    img_h = 32
    char_w = 16
    partition = 'tr'  # 'tr' / 'vl' / 'te'
    channels = 1
    batch_size = 8
    num_epochs = 200
    epochs_lr_decay = 100  # learning rate decay will be applied for last these many steps (should be <= num_epochs)
    resume_training = False
    start_epoch = 5

    train_gen_steps = 4  # generator weights to be updated after every specified number of steps
    grad_alpha = 1
    grad_balance = True

    data_file = f'./data/{dataset}_{partition}_data.pkl'
    lexicon_file = 'words_russian_5000.txt'
    lmdb_output = f'./data/{dataset}_{partition}_data'

    architecture = 'ScrabbleGAN'
    # Recognizer network
    r_ks = [3, 3, 3, 3, 3, 3, 2]
    r_pads = [1, 1, 1, 1, 1, 1, 0]
    r_fs = [64, 128, 256, 256, 512, 512, 512]

    # Generator and Discriminator networks
    # arch[g_resolution] defines the architecture to be selected
    # arch[16] has been added in BigGAN.py with parameters as specified in the paper
    resolution = 16
    bn_linear = 'SN'
    g_shared = False

    g_lr = 2e-4
    d_lr = 2e-4
    r_lr = 2e-4
    g_betas = [0., 0.999]
    d_betas = [0., 0.999]
    r_betas = [0., 0.999]
    g_loss_fn = 'HingeLoss'
    d_loss_fn = 'HingeLoss'
    r_loss_fn = 'CTCLoss'

    # Noise vector
    z_dim = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Config:
    # arch[img_h] defines the architecture to be selected
    # imh_h and char_width should be in: 32x16, 64x32, 128x64
    dataset = "Ru_Sber"
    
    img_h = 128
    char_w = 64
    channels = 3
    lexicon_file = None

    batch_size = 8
    num_epochs = 1000
    epochs_lr_decay = 100  # learning rate decay will be applied for last these many steps (should be <= num_epochs)

    train_gen_steps = 4  # generator weights to be updated after every specified number of steps
    grad_alpha = 1
    grad_balance = True

    architecture = 'ScrabbleGAN_2'

    # Generator and Discriminator networks
    bn_linear = 'SN'
    g_shared = False

    g_lr = 2e-4
    d_lr = 2e-4
    r_lr = 2e-4
    g_betas = [0., 0.999]
    d_betas = [0., 0.999]
    r_betas = [0., 0.999]
    g_loss_fn = 'HingeLoss'
    d_loss_fn = 'HingeLoss'
    r_loss_fn = 'CTCLoss'

    # Noise vector
    z_dim = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')