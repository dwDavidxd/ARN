import yaml
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn
from utils.model import LoadModel
from utils.util import ae_loss, calc_gradient_penalty, save_image

from utils.dataload import DatasetNPY # , DatasetIMG
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import math
from os.path import join

# Experiment Setting
# Reference: https://github.com/Alexander-H-Liu/UFDN/blob/master/train_face.py
cudnn.benchmark = True
config_path = './config/adver.yaml'
conf = yaml.load(open(config_path,'r'), Loader=yaml.FullLoader)
exp_name = conf['exp_setting']['exp_name']
img_size = conf['exp_setting']['img_size']
img_depth = conf['exp_setting']['img_depth']

trainer_conf = conf['trainer']

if trainer_conf['save_checkpoint']:
    model_path = conf['exp_setting']['checkpoint_dir'] + exp_name+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = model_path+'{}'

'''
if trainer_conf['save_log'] or trainer_conf['save_fig']:
    if os.path.exists(conf['exp_setting']['log_dir']+exp_name):
        shutil.rmtree(conf['exp_setting']['log_dir']+exp_name)
    writer = SummaryWriter(conf['exp_setting']['log_dir']+exp_name)
'''

# Random seed
np.random.seed(conf['exp_setting']['seed'])
_ = torch.manual_seed(conf['exp_setting']['seed'])

# Load dataset
nat_set = conf['exp_setting']['natural']
adv1_set = conf['exp_setting']['adv1']
adv2_set = conf['exp_setting']['adv2']
label_dirs = conf['exp_setting']['label_dirs']

test_nat_set = conf['exp_setting']['test_natural']
test_adv1_set = conf['exp_setting']['test_adv1']
test_adv2_set = conf['exp_setting']['test_adv2']
test_label_dirs = conf['exp_setting']['test_label_dirs']
batch_size = conf['trainer']['batch_size']


# trans = transforms.Compose([transforms.Resize(size=[28, 28]),transforms.ToTensor()])
trans = transforms.ToTensor()

# Load npy data
dataset = DatasetNPY(nat_dirs=nat_set, adv1_dirs=adv1_set, adv2_dirs=adv2_set, transform=trans)
'''
# Load png data
dataset = DatasetIMG(nat_dirs=nat_set, adv1_dirs=adv1_set, adv2_dirs=adv2_set, label_dirs=label_dirs,
                         transform=trans)
'''

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

# Test image
dataset_test = DatasetNPY(nat_dirs=test_nat_set, adv1_dirs=test_adv1_set, adv2_dirs=test_adv2_set, transform=trans)

data_loader_test =DataLoader(dataset_test, batch_size=10, shuffle=False, drop_last=False)

# Load Model
enc_dim = conf['model']['autoencoder']['encoder'][-1][1]
code_dim = conf['model']['autoencoder']['code_dim']
ae_learning_rate = conf['model']['autoencoder']['lr']
ae_betas = tuple(conf['model']['autoencoder']['betas'])
da_learning_rate = conf['model']['D_attack']['lr']
da_betas = tuple(conf['model']['D_attack']['betas'])
dp_learning_rate = conf['model']['D_pix']['lr']
dp_betas = tuple(conf['model']['D_pix']['betas'])

ae = LoadModel('autoencoder',conf['model']['autoencoder'],img_size,img_depth)
d_att = LoadModel('nn',conf['model']['D_attack'],img_size,enc_dim)
d_pix = LoadModel('cnn',conf['model']['D_pix'],img_size,img_depth)

reconstruct_loss = torch.nn.MSELoss()
clf_loss = nn.BCEWithLogitsLoss()

# Use cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
ae = ae.cuda()
ae = torch.nn.DataParallel(ae)
d_att = d_att.cuda()
d_att = torch.nn.DataParallel(d_att)
d_pix = d_pix.cuda()
d_pix = torch.nn.DataParallel(d_pix)

reconstruct_loss = reconstruct_loss.cuda()
clf_loss = clf_loss.cuda()
mse_loss = nn.MSELoss().cuda()
loss_bce = nn.BCEWithLogitsLoss().cuda()

# Optmizer
opt_ae = optim.Adam(list(ae.parameters()), lr=ae_learning_rate, betas=ae_betas)
opt_da = optim.Adam(list(d_att.parameters()), lr=da_learning_rate, betas=da_betas)
opt_dp = optim.Adam(list(d_pix.parameters()), lr=dp_learning_rate, betas=dp_betas)

ms = [int(1. / 4 * (trainer_conf['total_step']/100)), int(2. / 4 * (trainer_conf['total_step']/100))]

scheduler_ae = MultiStepLR(opt_ae, milestones=ms, gamma=0.1)
scheduler_da = MultiStepLR(opt_da, milestones=ms, gamma=0.5)
scheduler_dp = MultiStepLR(opt_dp, milestones=ms, gamma=0.5)

# Training

ae.train()
d_att.train()
d_pix.train()

# Domain code setting
code = np.concatenate([np.repeat(np.array([[*([1]),
                                            *([0]),
                                            *([0])]]),batch_size,axis=0),
                       np.repeat(np.array([[*([0]),
                                            *([1]),
                                            *([0])]]),batch_size,axis=0),
                       np.repeat(np.array([[*([0]),
                                            *([0]),
                                            *([1])]]),batch_size,axis=0)],
                       axis=0)

code = torch.FloatTensor(code)

trans_code = None

# Loss weight setting
loss_lambda = {}
for k in trainer_conf['lambda'].keys():
    init = trainer_conf['lambda'][k]['init']
    loss_lambda[k] = init

# Training
global_step = 0

while global_step < trainer_conf['total_step']:

    for nat, advu1, advu2 in data_loader:
        print('\repoch-' + str(math.floor(global_step / (math.ceil(60000/batch_size)))) + ':' + str(global_step % 200), end='')
        nat = nat.repeat(1, 3, 1, 1)
        advu1 = advu1.repeat(1, 3, 1, 1)
        advu2 = advu2.repeat(1, 3, 1, 1)
        input = torch.cat((nat, advu1, advu2), dim=0)
        input = (input * 2 - 1).cuda()
        target = torch.cat((nat, nat, nat), dim=0)
        target = (target * 2 - 1).cuda()

        code = code.cuda()
        invert_code = 0 * code + 1/3

        # Train Attack Discriminator
        opt_da.zero_grad()

        enc_x = ae(input, return_enc=True).detach()
        code_pred = d_att(enc_x)

        df_loss = clf_loss(code_pred, code)
        df_loss.backward()

        opt_da.step()

        # Train Pixel Discriminator
        opt_dp.zero_grad()

        pix_real_pred = d_pix(target)

        fake_data = ae(input, insert_attrs=trans_code)[0].detach()
        pix_fake_pred = d_pix(fake_data)

        t_real = torch.ones((batch_size * 3, 1)).cuda()
        t_fake = torch.zeros((batch_size * 3, 1)).cuda()

        gp = loss_lambda['gp'] * calc_gradient_penalty(d_pix, input.data, fake_data.data)

        pix_real_pred = loss_bce(pix_real_pred, t_real)
        pix_fake_pred = loss_bce(pix_fake_pred, t_fake)
        d_pix_loss = pix_real_pred + pix_fake_pred + gp
        d_pix_loss.backward()

        opt_dp.step()

        # Train VAE
        opt_ae.zero_grad()

        # Pixel-wise Phase
        fake_data, mu, logvar = ae(input, insert_attrs=trans_code)
        loss_mse, nor = ae_loss(fake_data, target, mu, logvar, reconstruct_loss)
        pixel_loss = loss_mse * loss_lambda['mse_for'] + nor * loss_lambda['nor']

        pixel_loss.backward()

        # Attack-invariant feature space adversarial Phase
        enc_x = ae(input, return_enc=True)
        domain_pred = d_att(enc_x)
        adv_code_loss = clf_loss(domain_pred, invert_code)
        att_loss = loss_lambda['attack_invariant'] * adv_code_loss

        att_loss.backward()

        # Pixel-wise adversarial Phase
        enc_x = ae(input, return_enc=True).detach()

        fake_data = ae.module.decode(enc_x, trans_code)

        adv_pix_loss = d_pix(fake_data)
        t_real = torch.ones((batch_size * 3, 1)).cuda()
        adv_pix_loss = loss_bce(adv_pix_loss, t_real)

        pixel_loss = loss_lambda['pix_adv'] * adv_pix_loss
        pixel_loss.backward()

        opt_ae.step()

        # End of step
        print('Step', global_step, end='\r', flush=True)
        global_step += 1

        # Save model
        if global_step % trainer_conf['checkpoint_step'] == 0 and trainer_conf['save_checkpoint'] and not trainer_conf[
            'save_best_only']:
            torch.save(ae.module.state_dict(), model_path.format(global_step) + '.ae')

            ae.eval()

            # Generate
            for nat, advu1, advu2 in data_loader_test:
                nat = nat.repeat(1, 3, 1, 1)
                advu1 = advu1.repeat(1, 3, 1, 1)
                advu2 = advu2.repeat(1, 3, 1, 1)
                input = torch.cat((nat, advu1, advu2), dim=0)
                input = (input * 2 - 1).cuda()
                break

            fake_data, _, _ = ae(input, insert_attrs=trans_code)
            fake_data = (fake_data + 1)/2
            if trainer_conf['save_fig']:
                save_image(fake_data, join('./checkpoint', 'test_restore_{}.png'.format(global_step)), nrow=3)

            ae.train()

    scheduler_ae.step()
    scheduler_da.step()
    scheduler_dp.step()
