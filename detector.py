import shutil
import cv2
import dlib
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import torchvision.transforms as transforms
from utils.load import *
from utils.process import *
from NIR_ISL2021.location import get_edge
from NIR_ISL2021.models import EfficientUNet
# from torchinfo import summary


def viz(module, grad_input, grad_output):
    global heatmap
    x = grad_output[0].cpu().numpy()
    # x = np.mean(x, axis=0)

    x = np.maximum(x[-1], 0)
    x /= np.max(x)
    x *= 255
    x = x.astype(np.uint8)
    heatmap = cv2.applyColorMap(cv2.resize(x,(320, 280)), cv2.COLORMAP_JET)


def main():
    num = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = EfficientUNet(num_classes=3).to(device)
    net = torch.nn.DataParallel(net, device_ids=[0])
    state_dict = torch.load(os.path.join(test_args['checkpoints_path'], 'for_mask.pth'), map_location=torch.device('cpu'))
    state_dict["module.heatmap4.loc.0.weight"] = state_dict.pop('module.loc4.loc.0.weight')
    state_dict["module.heatmap3.loc.0.weight"] = state_dict.pop('module.loc3.loc.0.weight')
    state_dict["module.heatmap2.loc.0.weight"] = state_dict.pop('module.loc2.loc.0.weight')
    state_dict["module.heatmap4.loc.0.bias"] = state_dict.pop('module.loc4.loc.0.bias')
    state_dict["module.heatmap3.loc.0.bias"] = state_dict.pop('module.loc3.loc.0.bias')
    state_dict["module.heatmap2.loc.0.bias"] = state_dict.pop('module.loc2.loc.0.bias')
    net.load_state_dict(state_dict)
    net.eval()
    net.module.dec1.register_forward_hook(viz)

    file_list = glob.glob(os.path.join(config['input_path'], '*.png'))
    if len(file_list) == 0:
        print('No files at given input path.')
        exit(-1)

    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])

    face_detector, landmark_Predictor = load_facedetector(config)

    for input_file in tqdm(file_list):
        # load image
        img = cv2.imread(input_file)

        if img is None or img is False:
            print("Could not open image file: %s" % input_file)
            continue

        img_eye_net_resize, img_eye_net_mask_resize, img_eye, img_eye_mask = get_crops_eye(face_detector, landmark_Predictor, img, input_file)

        if not len(img_eye_net_resize):
            continue

        # TODO obtain luminance difference
        iris_l, iris_l_mask, num_l = segment_iris(img_eye[0], img_eye_mask[0])
        iris_r, iris_r_mask, num_r = segment_iris(img_eye[1], img_eye_mask[1])

        if (num_l+num_r)/2 <= 90:
            # TODO the enhanced red channel
            eye_GrayRed_l = (img_eye_net_resize[0][..., 2] - img_eye_net_resize[0][..., 1] * 0.587 - img_eye_net_resize[0][..., 0] * 0.114) / 0.299
            eye_GrayRed_l = np.clip(eye_GrayRed_l, 0, 255)
            eye_GrayRed_l = cv2.cvtColor(eye_GrayRed_l.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            eye_GrayRed_r = (img_eye_net_resize[1][..., 2] - img_eye_net_resize[1][..., 1] * 0.587 - img_eye_net_resize[1][..., 0] * 0.114) / 0.299
            eye_GrayRed_r = np.clip(eye_GrayRed_r, 0, 255)
            eye_GrayRed_r = cv2.cvtColor(eye_GrayRed_r.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
            eye_GrayRed_l = cv2.filter2D(eye_GrayRed_l, -1, kernel=kernel)
            eye_GrayRed_r = cv2.filter2D(eye_GrayRed_r, -1, kernel=kernel)

            t = transforms.Compose([transforms.ToTensor()])
            eye_GrayRed_l = t(eye_GrayRed_l).unsqueeze(0).to(device)
            eye_GrayRed_r = t(eye_GrayRed_r).unsqueeze(0).to(device)
        else:
            # TODO EYE TELL ALL
            t = transforms.Compose([transforms.ToTensor()])
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
            eye_GrayRed_l = cv2.filter2D(img_eye_net_resize[0], -1, kernel=kernel)
            eye_GrayRed_r = cv2.filter2D(img_eye_net_resize[1], -1, kernel=kernel)
            eye_GrayRed_l = t(eye_GrayRed_l).unsqueeze(0).to(device)
            eye_GrayRed_r = t(eye_GrayRed_r).unsqueeze(0).to(device)

        with torch.no_grad():
            output_left = net(eye_GrayRed_l)
            output_right = net(eye_GrayRed_r)

        pred_pupil_mask_left = output_left['pred_pupil_mask']
        pred_pupil_mask_right = output_right['pred_pupil_mask']
        bIou_l = get_fit(pred_pupil_mask_left)
        bIou_r = get_fit(pred_pupil_mask_right)

        # TODO result
        i, j = os.path.splitext(input_file.split('\\')[-1])
        origin_imgl_Contour = get_draw_img(pred_pupil_mask_left, img_eye_net_resize[0])
        origin_imgr_Contour = get_draw_img(pred_pupil_mask_right, img_eye_net_resize[1])
        cv2.putText(origin_imgl_Contour, '{}'.format(f'{bIou_l:.2f}'), (0, 260), cv2.FONT_HERSHEY_TRIPLEX, 1.0,(0,0,255), 2)
        cv2.putText(origin_imgr_Contour, '{}'.format(f'{bIou_r:.2f}'), (0, 260), cv2.FONT_HERSHEY_TRIPLEX, 1.0,(0,0,255), 2)

        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(origin_imgl_Contour[...,::-1])
        num+=1
        plt.savefig('{}/{}_iris_final.png'.format(config['output_path'], i + '_{}'.format(num) + j), dpi=800, bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(origin_imgr_Contour[...,::-1])
        num+=1
        plt.savefig('{}/{}_iris_final.png'.format(config['output_path'],  i + '_{}'.format(num) + j), dpi=800, bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    test_args = {
        'dataset_name': 'CASIA-Iris-Africa',
        'checkpoints_path': r'./'
    }
    config = {
        'input_path': r'./face_data',
        'output_path': r'./heatmap1',
        'facedetector_path': r'./shape_predictor_68_face_landmarks.dat',
    }
    main()
