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
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
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
    #subject_list = os.listdir(dir_name)
    subject_list = ['003', '010', '011', '014', '016', '017', '018', '019', '021', '025', '027', '028', '029', '037']
    extension =".mp4"
    txt_dir = dir_name + 'test_label_txt2/wh/'
    scenario_list = ["_glasses_mix", "_nightglasses_mix", "_nightnoglasses_mix", "_noglasses_mix", "_sunglasses_mix"]
    txt_list = ['_glasses_mixing_drowsiness', '_nightglasses_mixing_drowsiness',
                '_night_noglasses_mixing_drowsiness', '_noglasses_mixing_drowsiness',
                '_sunglasses_mixing_drowsiness']

    save_dir = "D:/Dataset/NTHU-DDD-gc/Testing_Dataset2/"

    """for subject in subject_list:
        for index, scenario in enumerate(scenario_list):
            print(dir_name+subject+scenario+extension)
            cap = cv2.VideoCapture(dir_name+subject+scenario+extension)"""
    print("D:/Dataset/NTHU-DDD/Testing_Dataset/021_sunglasses_mix.mp4")
    cap = cv2.VideoCapture("D:/Dataset/NTHU-DDD/Testing_Dataset/021_sunglasses_mix.mp4")

    txt_num = 0
    frame_num = 0
    txt = []
    #label = open(txt_dir + subject + txt_list[index] + '.txt', 'r')
    label = open('D:/Dataset/NTHU-DDD/Testing_Dataset/test_label_txt2/wh/021_sunglasses_mixing_drowsiness.txt', 'r')
    label = list(label)
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
            height, width, channel = 64, 64, 3
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

                # warping
                facedstPoint = np.array([[300, 190], [380, 190], [340, 250]], dtype=np.float32)
                Rect_Point = np.array(
                    [[b[5], b[6]], [b[7], b[8]],
                     [b[13] - (b[13] - b[11]) / 2, b[14] - (b[14] - b[12]) / 2]],
                    dtype=np.float32)
                rect_matrix = cv2.getAffineTransform(Rect_Point, facedstPoint)
                fixed_face = cv2.warpAffine(gamma_contrast_norm_img, rect_matrix,
                                            (frame.shape[1], frame.shape[0]))  # width. height

                # left eye
                l_start_x = 260
                l_start_y = 170
                l_end_x = 330
                l_end_y = 210
                # cv2.rectangle(fixed_face, (l_start_x, l_start_y), (l_end_x, l_end_y), (0, 255, 0), 2)

                # warping
                left_eye_Point = np.array(
                    [[l_start_x, l_start_y], [l_end_x, l_start_y], [l_start_x, l_end_y],
                     [l_end_x, l_end_y]],
                    dtype=np.float32)
                left_matrix = cv2.getPerspectiveTransform(left_eye_Point, dstPoint)
                left_eye = cv2.warpPerspective(fixed_face, left_matrix, (width, height))

                # right eye
                r_start_x = 350
                r_start_y = 170
                r_end_x = 420
                r_end_y = 210
                # cv2.rectangle(fixed_face, (r_start_x, r_start_y), (r_end_x, r_end_y), (0, 255, 0), 2)

                # warping
                right_eye_Point = np.array(
                    [[r_start_x, r_start_y], [r_end_x, r_start_y], [r_start_x, r_end_y],
                     [r_end_x, r_end_y]],
                    dtype=np.float32)
                right_matrix = cv2.getPerspectiveTransform(right_eye_Point, dstPoint)
                right_eye = cv2.warpPerspective(fixed_face, right_matrix, (width, height))

                # mouth
                m_start_x = 300
                m_start_y = 225
                m_end_x = 380
                m_end_y = 280
                # cv2.rectangle(fixed_face, (m_start_x, m_start_y), (m_end_x, m_end_y), (0, 255, 0), 2)

                # warping
                mouth_Point = np.array(
                    [[m_start_x, m_start_y], [m_end_x, m_start_y], [m_start_x, m_end_y],
                     [m_end_x, m_end_y]],
                    dtype=np.float32)
                mouth_matrix = cv2.getPerspectiveTransform(mouth_Point, dstPoint)
                mouth = cv2.warpPerspective(fixed_face, mouth_matrix, (width, height))

                """# landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)"""

                """cv2.imshow("face", img_raw)
                # cv2.imshow("gamma_img", gamma_img)
                # cv2.imshow("contrast_norm", contrast_norm_img)
                cv2.imshow("gamma_contrast_norm", gamma_contrast_norm_img)
                # cv2.imshow("fixed face", fixed_face)
                cv2.imshow("left_eye", left_eye)
                cv2.imshow("right_eye", right_eye)
                cv2.imshow("mouth", mouth)"""

                txt.append(label[0][txt_num])

                # save image
                if args.save_image:

                    """if not os.path.exists(save_dir + subject + scenario + "/" + "left_eye/"):
                        os.makedirs(save_dir + subject + scenario + "/" + "left_eye/")
                    left_eye_path = save_dir + subject + scenario + "/" + "left_eye/" + str(
                        frame_num) + ".jpg"

                    if not os.path.exists(save_dir + subject + scenario + "/" + "right_eye/"):
                        os.makedirs(save_dir + subject + scenario + "/" + "right_eye/")
                    right_eye_path = save_dir + subject + "/" + "right_eye/" + str(
                        frame_num) + ".jpg"

                    if not os.path.exists(save_dir + subject + scenario + "/" + "mouth/"):
                        os.makedirs(save_dir + subject + scenario + "/" + "mouth/")
                    mouth_path = save_dir + subject + scenario + "/" + "mouth/" + str(
                        frame_num) + ".jpg" """
                    if not os.path.exists(save_dir + "021_sunglasses_mix/left_eye/"):
                        os.makedirs(save_dir + "021_sunglasses_mix/left_eye/")
                    left_eye_path = save_dir + "021_sunglasses_mix/left_eye/" + str(
                        frame_num) + ".jpg"

                    if not os.path.exists(save_dir + "021_sunglasses_mix/right_eye/"):
                        os.makedirs(save_dir + "021_sunglasses_mix/right_eye/")
                    right_eye_path = save_dir + "021_sunglasses_mix/right_eye/" + str(
                        frame_num) + ".jpg"

                    if not os.path.exists(save_dir + "021_sunglasses_mix/mouth/"):
                        os.makedirs(save_dir + "021_sunglasses_mix/mouth/")
                    mouth_path = save_dir + "021_sunglasses_mix/mouth/" + str(
                        frame_num) + ".jpg"

                    """if not os.path.exists(save_dir + subject + scenario + "/" + "face/"):
                        os.makedirs(save_dir + subject + scenario + "/" + "face/")
                    face_path = save_dir + subject + scenario + "/" + "face/" + str(
                        frame_num) + ".jpg" """

                    frame_num += 1

                    cv2.imwrite(left_eye_path, left_eye)
                    cv2.imwrite(right_eye_path, right_eye)
                    cv2.imwrite(mouth_path, mouth)
                    #cv2.imwrite(face_path, contrast_norm_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                break
            txt_num += 1
        else:
            break
    txt = np.array(txt, dtype=int)
    print(txt)
    #np.savetxt(save_dir + subject + "/" + subject + txt_list[index], txt, fmt='%d', delimiter='', newline='')

    np.savetxt(save_dir + "021_sunglasses_mix/" + '021_sunglasses_mixing_drowsiness.txt', txt, fmt='%d', delimiter='', newline='')
    cap.release()



