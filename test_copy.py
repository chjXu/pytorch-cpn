import os
import os.path
import sys
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import cv2
import json

from my_config import cfg
from torchvision.transforms import transforms
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import *
from utils.transforms import *
from utils.imutils import im_to_numpy, im_to_torch
from networks import network 
from dataloader.mscocoMulti import MscocoMulti
from tqdm import tqdm

class Detect2d( object ):
    def __init__(self, show_image=False):
        self.show_image = show_image
        self.inp_res = cfg.data_shapes
        self.pixel_means = cfg.pixel_means
        self.bbox_extend_factor = cfg.bbox_extend_factor
        self.model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained=False)
        self.model = torch.nn.DataParallel(self.model).cuda()

        # load trainning weights
        checkpoint_file = os.path.join('checkpoint', 'epoch37checkpoint.pth.tar')
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

        # change to evaluation mode
        self.model.eval()
        print('testing...')

    def detect(self, dets):
        dets = [i for i in dets if i['category_id'] == 1]
        dets.sort(key=lambda x: (x['image_id'], x['score']), reverse=True)
        dump_results = self.test_net(dets)
        return dump_results

    def cropImage(self, img, bbox):
        height, width = self.inp_res[0], self.inp_res[1]
        bbox = np.array(bbox).reshape(4, ).astype(np.float32)
        add = max(img.shape[0], img.shape[1])
        mean_value = self.pixel_means
        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=mean_value.tolist())
        objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])
        bbox += add
        objcenter += add

        crop_width = (bbox[2] - bbox[0]) * (1 + self.bbox_extend_factor[0] * 2)
        crop_height = (bbox[3] - bbox[1]) * (1 + self.bbox_extend_factor[1] * 2)

        if crop_height / height > crop_width / width:
            crop_size = crop_height
            min_shape = height
        else:
            crop_size = crop_width
            min_shape = width

        crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
        crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

        min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
        max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
        min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
        max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)

        x_ratio = float(width) / (max_x - min_x)
        y_ratio = float(height) / (max_y - min_y)

        img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))
        details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add]).astype(np.float)

        return img, details



    def test_net(self, dets):
        # create model
        #print(dets[0])
        # model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained=False)
        # model = torch.nn.DataParallel(model).cuda()
        #
        # # load trainning weights
        # checkpoint_file = os.path.join('checkpoint', 'CPN50_384x288.pth.tar')
        # checkpoint = torch.load(checkpoint_file)
        # model.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
        #
        # # change to evaluation mode
        # model.eval()
        # print('testing...')

        det_range = [0, len(dets)]  # three bboxs
        full_result = []

        start_time = time.time()

        for i in dets:
            img = i['data']
            #print(img)
            box = i['bbox']
            #print(data.shape)
            #data to pil_image
            #pil_image = Image.fromarray(data)

            transform1 = transforms.Compose([
                transforms.Resize(cfg.data_shapes),
                transforms.ToTensor()
            ])

            #image to tensor, and data shape should be [128, 3, 384, 288]
            #data = transform1(pil_image).unsqueeze(0)   #extend dimension to [1, 3, 384, 288]
            #print(data.shape)
            #inputs tensor type

            img, bbox = self.cropImage(img, box)
            # img = im_to_torch(img)
            # img = color_normalize(img, self.pixel_means)

            #print(img.shape)
            img = Image.fromarray(img)
            data = transform1(img).unsqueeze(0).float()
            #print(data.shape)

            with torch.no_grad():
                input_var = torch.autograd.Variable(data.cuda())

                # flip_inputs = input_var.clone()
                #
                # finp = im_to_numpy(flip_inputs[0])
                # finp = cv2.flip(finp, 1)
                # flip_inputs = im_to_torch(finp)
                # flip_input_var = torch.autograd.Variable(flip_inputs.cuda())

                # compute output
                global_outputs, refine_output = self.model(input_var)    #tensor
                score_map = refine_output.data.cpu()
                score_map = score_map.numpy()
                #print(score_map.shape)
                #(128, 17, 96, 72)

                # flip_global_outputs, flip_output = model(flip_input_var)
                # flip_score_map = flip_output.data.cpu()
                # flip_score_map = flip_score_map.numpy()
                #
                # for i, fscore in enumerate(flip_score_map):
                #     fscore = fscore.transpose((1, 2, 0))
                #     fscore = cv2.flip(fscore, 1)
                #     fscore = list(fscore.transpose((2, 0, 1)))
                #     for (q, w) in cfg.symmetry:
                #         fscore[q], fscore[w] = fscore[w], fscore[q]
                #     fscore = np.array(fscore)
                #     score_map[i] += fscore
                #     score_map[i] /= 2

                ids = i['image_id']
                det_scores = i['score']

                details = bbox
                single_result_dict = {}
                single_result = []

                img_h = 640
                img_w = 480
                single_map = score_map[0]
                #print(single_map.shape)

                # for m in single_map:
                #     h,w = np.unravel_index(m.argmax(), m.shape)
                #     print(h ,w)
                #     x = int(w*img_w/m.shape[1])
                #     y = int(h*img_h/m.shape[0])
                #     single_result.append(x)
                #     single_result.append(y)
                #     single_result.append(1)


                r0 = single_map.copy()
                r0 /= 255
                r0 += 0.5
                v_socre = np.zeros(17)
                for p in range(17):
                    single_map[p] /= np.amax(single_map[p])
                    border = 10
                    dr = np.zeros((cfg.output_shape[0] + 2*border, cfg.output_shape[1]+2*border))
                    dr[border:-border, border:-border] = single_map[p].copy()
                    dr = cv2.GaussianBlur(dr, (21, 21), 0)
                    lb = dr.argmax()
                    y, x = np.unravel_index(lb, dr.shape)
                    dr[y, x] = 0
                    lb = dr.argmax()
                    py, px = np.unravel_index(lb, dr.shape)
                    y -= border
                    x -= border
                    py -= border + y
                    px -= border + x
                    ln = (px ** 2 + py ** 2) ** 0.5
                    delta = 0.25
                    if ln > 1e-3:
                        x += delta * px / ln
                        y += delta * py / ln
                    x = max(0, min(x, cfg.output_shape[1] - 1))
                    y = max(0, min(y, cfg.output_shape[0] - 1))
                    resy = float((4 * y + 2) / cfg.data_shapes[0] * (details[3] - details[1]) + details[1])
                    resx = float((4 * x + 2) / cfg.data_shapes[1] * (details[2] - details[0]) + details[0])
                    v_socre[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])
                    single_result.append(resx)
                    single_result.append(resy)
                    single_result.append(1)

                if len(single_result) != 0:
                    single_result_dict['image_id'] = int(ids)
                    single_result_dict['category_id'] = 1
                    single_result_dict['keypoints'] = single_result
                    single_result_dict['score'] = float(det_scores) * v_socre.mean()
                    full_result.append(single_result_dict)
        #print(len(full_result))
        return full_result




if __name__ == '__main__':
    dets = [{'image_id': 0, 'category_id': 1, 'score': 0.9897885823249817,'bbox': [196,3,458,477]},
            {'image_id': 0, 'category_id': 1, 'score': 0.9997885823249817,'bbox': [332, 3, 582, 479]}
            #{'image_id': 0, 'category_id': 1, 'score': 0.9797885823249817,'bbox': [400, 116, 466, 297]}
            ]
    # [184, 194, 332, 531]
    #264, 33, 506, 380
    # trainp 8, 4, 145, 300
    #196,3,458,477  332, 3, 582, 479 video

    #video
    captures = cv2.VideoCapture(0)
    #oepncv version
    #(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    dector = Detect2d()

    cv2.namedWindow('Camera')
    while(True):
        start_time = time.time()
        ref, frame = captures.read()
        for i in range(len(dets)):
            bbox = dets[i]['bbox']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1, 8)

        for i in dets:
            i['data'] = frame #% i['image_id']
            #cv2.imshow('Camera', i['data'])

        #print(dector.detect(dets))
        # x = []
        # y = []
        # s = []
        for i in dector.detect(dets):
            points = np.array(i['keypoints']).reshape(17, 3).astype(np.float32)
            points = points.flatten()

            x = points[0::3]
            y = points[1::3]
            s = points[2::3]

            for p in range(x.size):
                cv2.circle(frame,(x[p],y[p]),4, (0,255,255), 3)
                    #cv2.imshow('Camera', frame)
                    #cv2.waitKey(3)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps = "%.2f FPS" % fps
        cv2.putText(frame, fps, (5,50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255),2)
        cv2.imshow('Camera', frame)
        c = cv2.waitKey(30) & 0xff
        if c==27:
            break

    captures.release()
    cv2.destroyAllWindows()

    # multi-bounding boxs(all right)
    # start_time = time.time()
    # for i in dets:
    #     i['data'] = cv2.imread('../images/img_%02d.jpg' % i['image_id'])  #i['data'] means a photo
    #     #print(type(i['data']))
    #     # cv2.rectangle(i['data'], (i['bbox'][0], i['bbox'][1]), (i['bbox'][2], i['bbox'][3]), (0, 0, 255), 1, 8)
    #     # cv2.imshow('image', i['data'])
    #     # cv2.waitKey(3000)
    #
    # dector = Detect2d()
    # #print(dector.detect(dets))
    # img = cv2.imread('../images/img_00.jpg')
    # while(1):
    #     for i in dector.detect(dets):
    #     #print(len(i['keypoints']))
    #         points = np.array(i['keypoints']).reshape(17,3).astype(np.float32)
    #         points = points.flatten()
    #
    #         x = points[0::3]
    #         y = points[1::3]
    #         s = points[2::3]
    #
    #         for p in range(x.size):
    #             cv2.circle(img,(x[p],y[p]),4, (255,255,0), 3)
    #             cv2.imshow('image', img)
    #             cv2.waitKey(3)
    #
    #     end_time = time.time()
    #     print(end_time-start_time)
    #     fps = 1 / (end_time - start_time)
    #     fps = "%.2f FPS" % fps
    #     cv2.putText(img, fps, (5,50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255),2)


            # plt.figure()
            # c = (np.random.random((1,3)) * 0.6 + 0.4).tolist()[0]
            # plt.plot(x[s>0], y[s>0], 'o', markersize=10, markerfacecolor=c, markeredgecolor='k', markeredgewidth=2)
            # img = Image.open('../images/img_00.jpg')
            # plt.imshow(img)
            # plt.show()
            # cv2.waitKey(3)



    # parser = argparse.ArgumentParser(description='PyTorch CPN Test')
    # parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
    #                     help='number of data loading workers (default: 12)')
    # parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
    #                     help='number of GPU to use (default: 1)')
    # parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
    #                     help='path to load checkpoint (default: checkpoint)')
    # parser.add_argument('-f', '--flip', default=True, type=bool,
    #                     help='flip input image during test (default: True)')
    # parser.add_argument('-b', '--batch', default=128, type=int,
    #                     help='test batch size (default: 128)')
    # parser.add_argument('-t', '--test', default='CPN384x288', type=str,
    #                     help='using which checkpoint to be tested (default: CPN256x192')
    # parser.add_argument('-r', '--result', default='result', type=str,
    #                     help='path to save save result (default: result)')
    #main(parser.parse_args())
