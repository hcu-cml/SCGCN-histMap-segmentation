from __future__ import division

import sys
import os


from tools.model import load_model

from config.configs_kf import *
from lib.utils.visual import *

import torchvision.transforms as st
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import matplotlib

from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from tqdm import tqdm_notebook as tqdm

from tools.ckpt import *
import time
import itertools


output_path = os.path.join('./results/', 'results_paris_maps')

#precision =/ recall different results

def main():
    check_mkdir(output_path)
    nets = []

    net1 = get_net(ckpt1)  # MSCG-Net-R50
    # net2 = get_net(ckpt2)  # MSCG-Net-R101

    nets.append(net1)
    # nets.append(net2)


    test_files = get_real_test_list(bands=['RGB'])
    
    #change from 1 to 5
    tta_real_test(nets, stride=600, batch_size=1,
                  norm=False, window_size=(500, 500), labels=land_classes,
                  test_set=test_files, all=True)

def fusion_prediction(nets, image, scales, batch_size=1, num_class=4, wsize = (500, 500)):
    pred_all = np.zeros(image.shape[:2] + (num_class,))
    for scale_rate in scales:
        # print('scale rate: ', scale_rate)
        img = image.copy()
        img = scale(img, scale_rate)
        pred = np.zeros(img.shape[:2] + (num_class,))
        stride = img.shape[1]
        window_size = img.shape[:2]
        total = count_sliding_window(img, step=stride, window_size=wsize) // batch_size
        for i, coords in enumerate(
                tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                     leave=False)):

            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            imgs_flip = [patch[:, ::-1, :] for patch in image_patches]
            imgs_mirror = [patch[:, :, ::-1] for patch in image_patches]

            image_patches = np.concatenate((image_patches, imgs_flip, imgs_mirror), axis=0)
            image_patches = np.asarray(image_patches)
            image_patches = torch.from_numpy(image_patches).cuda()

            for net in nets:
                outs = net(image_patches) #+ Fn.torch_rot270(net(Fn.torch_rot90(image_patches)))
                # outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                b, _, _, _ = outs.shape

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs[0:b // 3, :, :, :], coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

                for out, (x, y, w, h) in zip(outs[b // 3:2 * b // 3, :, :, :], coords):
                    out = out[:, ::-1, :]  # flip back
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

                for out, (x, y, w, h) in zip(outs[2 * b // 3: b, :, :, :], coords):
                    out = out[:, :, ::-1]  # mirror back
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

                del (outs)

        pred_all += scale(pred, 1.0 / scale_rate)

    return pred_all



def tta_real_test(nets, all=False, labels=land_classes, norm=False,
             test_set=None, stride=600, batch_size=7, window_size=(500, 500)):
    # test_images, test_labels = (loadtestimg(test_set))
    test_files = (loadtestimg(test_set))
    # gts = (loadgt(test_set))
    idlist=(loadids(test_set))

    all_preds = []
    all_gts = []
    num_class = len(labels)
    ids = []

    total_ids = 0

    for k in test_set[IDS].keys():
        total_ids += len(test_set[IDS][k])

    for img, id in tqdm(zip(test_files, idlist), total=total_ids, leave=False):


        if id[:2] == "._":
            id = id[2:]

        img = np.asarray(img, dtype='float32')
        img = st.ToTensor()(img)
        img = img / 255.0
        if norm:
            img = st.Normalize(*mean_std)(img)
        img = img.cpu().numpy().transpose((1, 2, 0))


        stime = time.time()


        with torch.no_grad():
            pred = fusion_prediction(nets, image=img, scales=[1.0],
                                     batch_size=batch_size, num_class=num_class, wsize=window_size)

        print('inference cost time: ', time.time() - stime)


        pred = np.argmax(pred, axis=-1)

        pred_image = np.zeros((500, 500, 3), dtype=int)


        filename = './{}.png'.format(id)

        palette_land = {
            0 : (0, 0, 0),        # background
            1 : (0, 0, 255),    # cloud_shadow
            2 : (255, 0, 255),    # double_plant
            3 : (0, 255, 255),      # planter_skip
            4 : (255, 255, 255),      # standing_water
        }

        for i in range(500):
            for k in range(500):
                pred_image[i,k,0] = palette_land[pred[i,k]][2]
                pred_image[i,k,1] = palette_land[pred[i,k]][1]
                pred_image[i,k,2] = palette_land[pred[i,k]][0]
                print(pred[i,k])

        cv2.imwrite(os.path.join("/home_domuser/s6luarzo/masterarbeit-arzoumanidis/MSCG-Net_paris/results/results_style_transfer/", filename), pred_image)


        all_preds.append(pred)

        gt = cv2.imread('/home_domuser/s6luarzo/paris_500/test/gt/' + id + '.png', cv2.IMREAD_GRAYSCALE)

        all_gts.append(gt)

        ids.append(id)


    accuracy, cm = metrics(np.concatenate([p.ravel() for p in all_preds]),
    np.concatenate([p.ravel() for p in all_gts]).ravel(), label_values=labels)
        
    M, N = cm.shape
    tp = np.zeros(M, dtype=np.uint)
    fp = np.zeros(M, dtype=np.uint)
    fn = np.zeros(M, dtype=np.uint)

    for i in range(M):
        tp[i] = cm[i, i]
        fn[i] = np.sum(cm[:, i]) - tp[i]
        fp[i] = np.sum(cm[i, :]) - tp[i]

    precision = tp / (tp + fp)  
    recall = tp / (tp + fn)

    precision_total = np.nanmean(precision)
    recall_total = np.nanmean(recall)


    precision = precision_score(np.concatenate([p.ravel() for p in all_gts]).ravel(), np.concatenate([p.ravel() for p in all_preds]), average=None)
    recall = recall_score(np.concatenate([p.ravel() for p in all_gts]).ravel(), np.concatenate([p.ravel() for p in all_preds]), average=None)
    mIoU = jaccard_score(np.concatenate([p.ravel() for p in all_gts]).ravel(), np.concatenate([p.ravel() for p in all_preds]), average='macro')
    IoU = jaccard_score(np.concatenate([p.ravel() for p in all_gts]).ravel(), np.concatenate([p.ravel() for p in all_preds]), average=None)

    print("----------------------------------------------")
    print("Precision Frame: " + str(precision[0]))
    print("Precision Water: " + str(precision[1]))
    print("Precision Blocks: " + str(precision[2]))
    print("Precision Non-build: " + str(precision[3]))
    print("Precision Road network: " + str(precision[4]))
    print("----------------------------------------------")
    print("Recall Frame: " + str(recall[0]))
    print("Recall Water: " + str(recall[1]))
    print("Recall Blocks: " + str(recall[2]))
    print("Recall Non-build: " + str(recall[3]))
    print("Recall Road network: " + str(recall[4]))
    print("----------------------------------------------")
    print("IoU Frame: " + str(IoU[0]))
    print("IoU Water: " + str(IoU[1]))
    print("IoU Blocks: " + str(IoU[2]))
    print("IoU Non-build: " + str(IoU[3]))
    print("IoU Road network: " + str(IoU[4]))
    print("----------------------------------------------")
    print("Precision: " + str(precision_total))
    print("Recall: " + str(recall_total))
    print("mIoU: " + str(mIoU))
    print("----------------------------------------------")

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmap = plt.colormaps['magma']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Frame', 'Water', 'Blocks', 'Non-build', 'Road Net.'])
    

    fontSize = 11
    font = {'size': fontSize, 'family': 'serif'}
    matplotlib.rc('font', **font)
    matplotlib.rc('text', usetex=True)

    disp.plot()
    plt.savefig('confusionsmatrix.pdf', bbox_inches='tight')





def metrics(predictions, gts, label_values=land_classes):

    print(np.shape(gts), "shape gt")
    print(np.shape(predictions), "shape pred")

    cm = confusion_matrix(gts, predictions, labels=range(len(label_values)))

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except BaseException:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: " + str(kappa))
    return accuracy, cm


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    # print(top.shape[0])
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    print('total number of sliding windows: ', c)
    return c


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    main()
