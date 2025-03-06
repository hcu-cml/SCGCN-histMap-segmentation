
import sys 
import os

from tools.model import *
from config.configs_kf import *

#test

ckpt1 = {
    'net': 'MSCG-Rx50',
    'data': 'Agriculture',
    'bands': ['RGB'],
    'nodes': (64,64),
    'snapshot': '/home_domuser/s6luarzo/masterarbeit-arzoumanidis/MSCG-Net_paris/ckpt/beste_b9/epoch_429_loss_0.82157_acc_0.85444_acc-cls_0.87224_mean-iu_0.71597_fwavacc_0.75489_f1_0.82678_lr_0.79975_precision_0.87224_recall_0.0000199939.pth'
}




def get_net(ckpt=ckpt1):
    net = load_model(name=ckpt['net'],
                     classes=5,
                     node_size=ckpt['nodes'])

    net.load_state_dict(torch.load(ckpt['snapshot']), strict=False)
    net.cuda()
    net.eval()
    return net


def loadtestimg(test_files):

    id_dict = test_files[IDS]
    image_files = test_files[IMG]
    # mask_files = test_files[GT]

    for key in id_dict.keys():
        for id in id_dict[key]:
            if len(image_files) > 1:
                imgs = []
                for i in range(len(image_files)):
                    filename = image_files[i].format(id)
                    path, _ = os.path.split(filename)
                    if path[-3:] == 'nir':
                        # img = imload(filename, gray=True)
                        img = np.asarray(Image.open(filename), dtype='uint8')
                        img = np.expand_dims(img, 2)

                        imgs.append(img)
                    else:
                        img = imload(filename)
                        imgs.append(img)
                image = np.concatenate(imgs, 2)
            else:
                filename = image_files[0].format(id)
                path, _ = os.path.split(filename)
                if path[-3:] == 'nir':
                    # image = imload(filename, gray=True)
                    image = np.asarray(Image.open(filename), dtype='uint8')
                    image = np.expand_dims(image, 2)
                else:
                    print(filename)
                    image = imload(filename)
            # label = np.asarray(Image.open(mask_files.format(id)), dtype='uint8')

            yield image


def loadids(test_files):
    id_dict = test_files[IDS]

    for key in id_dict.keys():
        for id in id_dict[key]:
            yield id


def loadgt(test_files):
    id_dict = test_files[IDS]
    mask_files = test_files[GT]
    for key in id_dict.keys():
        for id in id_dict[key]:
            label = np.asarray(Image.open(mask_files.format(id)), dtype='uint8')
            yield label
