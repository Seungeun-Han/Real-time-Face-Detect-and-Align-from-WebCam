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
import dlib
import models

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

    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()

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
        print('net forward time: {:.4f}'.format(time.time() - tic))

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
        height, width, channel = 120, 160, 3
        dstPoint = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
        # show image
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            face_width = b[2] - b[0]
            face_height = b[3] - b[1]

            # left eye
            l_start_x = b[5] - int(0.25 * face_width)
            l_start_y = b[6] - int(0.1 * face_height)
            l_end_x = b[5] + int(0.15 * face_width)
            l_end_y = b[6] + int(0.1 * face_height)
            cv2.rectangle(img_raw, (l_start_x, l_start_y), (l_end_x, l_end_y), (0, 255, 0), 2)

            # warping
            left_eye_Point = np.array([[l_start_x, l_start_y], [l_end_x, l_start_y], [l_start_x, l_end_y], [l_end_x, l_end_y]], dtype=np.float32)
            left_matrix = cv2.getPerspectiveTransform(left_eye_Point, dstPoint)
            left_eye = cv2.warpPerspective(frame, left_matrix, (width, height))

            # right eye
            r_start_x = b[7] - int(0.15 * (b[2] - b[0]))
            r_start_y = b[8] - int(0.1 * (b[3] - b[1]))
            r_end_x = b[7] + int(0.25 * (b[2] - b[0]))
            r_end_y = b[8] + int(0.1 * (b[3] - b[1]))
            cv2.rectangle(img_raw, (r_start_x, r_start_y), (r_end_x, r_end_y), (0, 255, 0), 2)

            # warping
            right_eye_Point = np.array(
                [[r_start_x, r_start_y], [r_end_x, r_start_y], [r_start_x, r_end_y], [r_end_x, r_end_y]],
                dtype=np.float32)
            right_matrix = cv2.getPerspectiveTransform(right_eye_Point, dstPoint)
            right_eye = cv2.warpPerspective(frame, right_matrix, (width, height))

            # mouth
            m_start_x = b[11] - 5
            m_start_y = b[12] - int(0.2 * (b[3] - b[1]))
            m_end_x = b[13] + 5
            m_end_y = b[14] + int(0.25 * (b[3] - b[1]))
            cv2.rectangle(img_raw, (m_start_x, m_start_y), (m_end_x, m_end_y), (0, 255, 0), 2)

            # warping
            mouth_Point = np.array(
                [[m_start_x, m_start_y], [m_end_x, m_start_y], [m_start_x, m_end_y], [m_end_x, m_end_y]],
                dtype=np.float32)
            mouth_matrix = cv2.getPerspectiveTransform(mouth_Point, dstPoint)
            mouth = cv2.warpPerspective(frame, mouth_matrix, (width, height))


            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            cv2.imshow("face", img_raw)
            cv2.imshow("left_eye", left_eye)
            cv2.imshow("right_eye", right_eye)
            cv2.imshow("mouth", mouth)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        """
        # save image
        if args.save_image:
            name = "test.jpg"
            cv2.imwrite(name, img_raw)
        """
"""
        # filter using vis_thres
        faces = []
        for b in dets:
            if b[4] > args.vis_thres:
                xmin, ymin, xmax, ymax, score = b[0], b[1], b[2], b[3], b[4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                bbox = [xmin, ymin, xmax, ymax, score]
                faces.append(bbox)

        if len(faces) == 0:
            print('NO face is detected!')
            continue

        height, width, _ = frame.shape
        for k, face in enumerate(faces):  # project11 참고
            x1, y1, x2, y2 = face[0], face[1], face[2], face[3]

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            size = int(min([w, h]) * 1.2)
            cx = x1 + w // 2
            cy = y1 + h // 2
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = box_utils.BBox(new_bbox)
            cropped = frame[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
            cropped_face = cv2.resize(cropped, (112, 112))

            if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
                continue
            test_face = cropped_face.copy()
            test_face = test_face / 255.0

            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape((1,) + test_face.shape)
            input = torch.from_numpy(test_face).float()
            input = torch.autograd.Variable(input)

            # start = time.time()
            if args.network == 'MobileFaceNet':
                landmark = net(input)[0].cpu().data.numpy()
            else:
                landmark = net(input).cpu().data.numpy()
            # end = time.time()
            # print('Time: {:.6f}s.'.format(end - start))

            landmark = landmark.reshape(-1, 2)

            landmark = new_bbox.reprojectLandmark(landmark)


            img_land = box_utils.drawLandmark(frame, bbox, landmark)
            cv2.imshow("sdf", img_land)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break"""


