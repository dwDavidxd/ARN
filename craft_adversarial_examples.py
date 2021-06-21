import matplotlib.pyplot as plt
import os
import torch
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader,get_mnist_train_loader
from advertorch_examples.utils import bchw2bhwc
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, DDNL2Attack, SinglePixelAttack, LocalSearchAttack, SpatialTransformAttack,L1PGDAttack
import numpy as np
import argparse
import torch.nn as nn
import pickle
import cv2


def main(args):
    plt.switch_backend('agg')

    torch.manual_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.exists(args.savepath_adv):
        os.makedirs(args.savepath_adv)

    if not os.path.exists(args.savepath_nat):
        os.makedirs(args.savepath_nat)

    filename = "mnist_lenet5_clntrained.pt"

    model = LeNet5()
    model.load_state_dict(
        torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    batch_size = args.batch_size

    # Training data
    loader = get_mnist_train_loader(batch_size=batch_size, shuffle=False)

    # Test data
    # loader = get_mnist_test_loader(batch_size=batch_size, shuffle=False)

    # Path to save adversarial examples
    path_adv = args.savepath_adv
    path_nat = args.savepath_nat

    cnt = 0
    acc_nat = 0
    acc_adv = 0
    data_number = 0
    label_true = []

    for nat_data, true_label in loader:
        data_number = data_number + nat_data.size(0)

        for x in true_label:
            label_true.append(x.item())

        nat_data, true_label = nat_data.to(device), true_label.to(device)

        # PGD
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.30,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

        '''
        # CW
        adversary = CarliniWagnerL2Attack(
            model, 10, clip_min=0.0, clip_max=1.0, max_iterations=500, confidence=1, initial_const=1, learning_rate=1e-2,
            binary_search_steps=4, targeted=False)
        '''

        '''
        # DDN
        adversary = DDNL2Attack(model, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True, levels=256, clip_min=0.0,
                                clip_max=1.0, targeted=False, loss_fn=None)
        '''

        '''
        # STA
        adversary = SpatialTransformAttack(
            model, 10, clip_min=0.0, clip_max=1.0, max_iterations=5000, search_steps=20, targeted=False)
        '''

        # Craft non-target adversarial examples
        adv = adversary.perturb(nat_data, true_label)

        # Predict natural examples
        pred_nat = predict_from_logits(model(nat_data))
        # Predict adversarial examples
        pred_adv = predict_from_logits(model(adv))

        num0 = 0
        num1 = 0

        for n in range(batch_size):
            if pred_nat[n] == true_label[n]:
                num0 += 1
        for n in range(batch_size):
            if pred_adv[n] == true_label[n]:
                num1 += 1

        print("Accuracy of natural examples:" + str(num0 / batch_size))
        print("Accuracy of adversarial examples:" + str(num1 / batch_size))

        acc_nat += num0
        acc_adv += num1

        # Save examples
        for n in range(batch_size):
            cnt += 1

            # Save natural examples
            img = bchw2bhwc(nat_data[n].detach().cpu().numpy())
            # (.npy)
            name = str(cnt) + '.npy'
            path = os.path.join(path_nat, name)
            np.save(path, img)

            '''
            # (.png)
            name = str(cnt) + '.png'
            path = os.path.join(path_adv, name)
            cv2.imwrite(path, img*255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            '''

            # Save adversarial examples
            img = bchw2bhwc(adv[n].detach().cpu().numpy())
            # (.npy)
            name = str(cnt) + '.npy'
            path = os.path.join(path_adv, name)
            np.save(path, img)

            '''
            # (.png)
            name = str(cnt) + '.png'
            path = os.path.join(path_adv, name)
            cv2.imwrite(path, img*255, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            '''

        print('Craft ' + str(cnt) + ' adversarial examples-------------')

    print("| Average: Natural Result\tAcc@1: %.2f%%" % (100. * acc_nat / data_number))
    print("| Average: Attack Result\tAcc@1: %.2f%%" % (100. * acc_adv / data_number))


    # save natural label
    pickle.dump(label_true, open(args.savepath_label, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack')
    parser.add_argument('--savepath_adv', default='./adv_example/train/adv1')
    parser.add_argument('--savepath_nat', default='./adv_example/train/nat')
    parser.add_argument('--savepath_label', default='./adv_example/train/label_true.pkl')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    parser.add_argument('--seed', default=0)

    args = parser.parse_args()

    main(args)




