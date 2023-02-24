from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import math

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1
    dir_name = "D:/Dataset/NTHU-DDD/Testing_Dataset/"
    subject_list = ['003', '010', '011', '014', '016', '017', '018', '019', '021', '025', '027', '028', '029', '037']
    scenario_list = ["_glasses_mix", "_nightglasses_mix", "_nightnoglasses_mix", "_noglasses_mix", "_sunglasses_mix"]
    extension = ".mp4"
    save_dir = "D:/Dataset/NTHU-DDD-gc/Testing_Dataset2/"

    """for subject in subject_list:
            for scenario in scenario_list:
                print(dir_name+subject+scenario+extension)
                cap = cv2.VideoCapture(dir_name+subject+scenario+extension)"""
    print("D:/Dataset/NTHU-DDD/Testing_Dataset/003_nightnoglasses_mix.mp4")
    cap = cv2.VideoCapture(1) # "D:/Dataset/NTHU-DDD/Testing_Dataset/003_nightnoglasses_mix.mp4"

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img = np.float32(frame)

            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            tic = time.time()
            loc, conf, landms = net(img)  # forward pass
            #print('net forward time: {:.4f}'.format(time.time() - tic))

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            img_raw = frame.copy()
            # img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
            height, width, channel = 256, 256, 3
            dstPoint = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)

            # gamma correction
            optical_gamma = math.log(np.mean(128) / 255) / math.log(np.mean(img_raw) / 255)
            img_raw = img_raw.astype(np.float)
            gamma_img = 255 * ((img_raw / 255) ** optical_gamma)
            img_raw = img_raw.astype(np.uint8)
            gamma_img = gamma_img.astype(np.uint8)

            # Contrast normalization
            #contrast_norm_img = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX)
            gamma_contrast_norm_img = cv2.normalize(gamma_img, None, 0, 255, cv2.NORM_MINMAX)

            # show image
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                #cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                #cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                face_width = b[2] - b[0]
                face_height = b[3] - b[1]

                facedstPoint = np.array([[250, 150], [378, 150], [378, 278]], dtype=np.float32)
                Rect_Point = np.array(
                    [[b[0], b[1]], [b[0] + face_width, b[1]], [b[2], b[3]]],
                    dtype=np.float32)
                rect_matrix = cv2.getAffineTransform(Rect_Point, facedstPoint)
                fixed_face = cv2.warpAffine(gamma_contrast_norm_img, rect_matrix,
                                            (frame.shape[1], frame.shape[0]))  # width. height

                # face
                start_x = 250
                start_y = 150
                end_x = 378
                end_y = 278
                # cv2.rectangle(fixed_face, (l_start_x, l_start_y), (l_end_x, l_end_y), (0, 255, 0), 2)

                # warping
                Point = np.array(
                    [[start_x, start_y], [end_x, start_y], [start_x, end_y], [end_x, end_y]],
                    dtype=np.float32)
                matrix = cv2.getPerspectiveTransform(Point, dstPoint)
                face = cv2.warpPerspective(fixed_face, matrix, (width, height))

                # save image
                if args.save_image:
                    if not os.path.exists(save_dir + "003_nightnoglasses_mix/face/"):
                        os.makedirs(save_dir + "003_nightnoglasses_mix/face/")
                    face_path = save_dir + "003_nightnoglasses_mix/face/" + str(
                        frame_num) + ".jpg"

                    frame_num += 1

                    cv2.imwrite(face_path, face)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cv2.imshow("face", face)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                break
        else:
            break
    cap.release()




