from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
import cv2
import heapq
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)

    parser.add_argument('--load_weights_folder', type=str,
                        help='path of a pretrained model to use',
                        )

    parser.add_argument('--test',
                        action='store_true',
                        help='if set, read images from a .txt file',
                        )

    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        default="lite-mono",
                        choices=[
                            "lite-mono",
                            "lite-mono-small",
                            "lite-mono-tiny",
                            "lite-mono-8m"])

    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.load_weights_folder is not None, \
        "You must specify the --load_weights_folder parameter"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    # extract the height and width of image that this model was trained with
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.LiteMono(model=args.model,
                                    height=feed_height,
                                    width=feed_width)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path) and not args.test:
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isfile(args.image_path) and args.test:
        gt_path = os.path.join('splits', 'eigen', "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        # reading images from .txt file
        paths = []
        with open(args.image_path) as f:
            filenames = f.readlines()
            for i in range(len(filenames)):
                filename = filenames[i]
                line = filename.split()
                folder = line[0]
                if len(line) == 3:
                    frame_index = int(line[1])
                    side = line[2]

                f_str = "{:010d}{}".format(frame_index, '.jpg')
                image_path = os.path.join(
                    'kitti_data',
                    folder,
                    "image_0{}/data".format(side_map[side]),
                    f_str)
                paths.append(image_path)

    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            # output_name = os.path.splitext(image_path)[0].split('/')[-1]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))


    print('-> Done!')
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     test_simple(args)
















# from __future__ import absolute_import, division, print_function
#
# import os
# import sys
# import glob
# import argparse
# import numpy as np
# import PIL.Image as pil
# import matplotlib as mpl
# import matplotlib.cm as cm
#
# import torch
# from torchvision import transforms, datasets
#
# import networks
# from layers import disp_to_depth
# import cv2
# import heapq
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Simple testing function for Lite-Mono models.')
#
#     parser.add_argument('--image_path', type=str,
#                         help='path to a test image or folder of images', required=True)
#
#     parser.add_argument('--load_weights_folder', type=str,
#                         help='path of a pretrained model to use',
#                         )
#
#     parser.add_argument('--test',
#                         action='store_true',
#                         help='if set, read images from a .txt file',
#                         )
#
#     parser.add_argument('--model', type=str,
#                         help='name of a pretrained model to use',
#                         default="lite-mono",
#                         choices=[
#                             "lite-mono",
#                             "lite-mono-small",
#                             "lite-mono-tiny",
#                             "lite-mono-8m"])
#
#     parser.add_argument('--ext', type=str,
#                         help='image extension to search for in folder', default="jpg")
#     parser.add_argument("--no_cuda",
#                         help='if set, disables CUDA',
#                         action='store_true')
#
#     return parser.parse_args()
#
#
# def test_simple(args):
#     """Function to predict for a single image or folder of images
#     """
#     assert args.load_weights_folder is not None, \
#         "You must specify the --load_weights_folder parameter"
#
#     if torch.cuda.is_available() and not args.no_cuda:
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     print("-> Loading model from ", args.load_weights_folder)
#     encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
#     decoder_path = os.path.join(args.load_weights_folder, "depth.pth")
#
#     encoder_dict = torch.load(encoder_path)
#     decoder_dict = torch.load(decoder_path)
#
#     # extract the height and width of image that this model was trained with
#     feed_height = encoder_dict['height']
#     feed_width = encoder_dict['width']
#
#     # LOADING PRETRAINED MODEL
#     print("   Loading pretrained encoder")
#     encoder = networks.LiteMono(model=args.model,
#                                     height=feed_height,
#                                     width=feed_width)
#
#     model_dict = encoder.state_dict()
#     encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
#
#     encoder.to(device)
#     encoder.eval()
#
#     print("   Loading pretrained decoder")
#     depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
#     depth_model_dict = depth_decoder.state_dict()
#     depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
#
#     depth_decoder.to(device)
#     depth_decoder.eval()
#
#     # FINDING INPUT IMAGES
#     if os.path.isfile(args.image_path) and not args.test:
#         # Only testing on a single image
#         paths = [args.image_path]
#         output_directory = os.path.dirname(args.image_path)
#     elif os.path.isfile(args.image_path) and args.test:
#         gt_path = os.path.join('splits', 'eigen', "gt_depths.npz")
#         gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
#
#         side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
#         # reading images from .txt file
#         paths = []
#         with open(args.image_path) as f:
#             filenames = f.readlines()
#             for i in range(len(filenames)):
#                 filename = filenames[i]
#                 line = filename.split()
#                 folder = line[0]
#                 if len(line) == 3:
#                     frame_index = int(line[1])
#                     side = line[2]
#
#                 f_str = "{:010d}{}".format(frame_index, '.jpg')
#                 image_path = os.path.join(
#                     'kitti_data',
#                     folder,
#                     "image_0{}/data".format(side_map[side]),
#                     f_str)
#                 paths.append(image_path)
#
#     elif os.path.isdir(args.image_path):
#         # Searching folder for images
#         paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
#         output_directory = args.image_path
#     else:
#         raise Exception("Can not find args.image_path: {}".format(args.image_path))
#
#     print("-> Predicting on {:d} test images".format(len(paths)))
#     cap = cv2.VideoCapture(0)  # 0 通常代表默认摄像头
#     if not cap.isOpened():
#         print("Error: Could not open camera")
#         exit()
#
#     print("-> Starting camera feed")
#
#     # PREDICTING ON EACH IMAGE IN TURN
#     with torch.no_grad():
#         for idx, image_path in enumerate(paths):
#
#             if image_path.endswith("_disp.jpg"):
#                 # don't try to predict disparity for a disparity image!
#                 continue
#
#             # Load image and preprocess
#             input_image = pil.open(image_path).convert('RGB')
#             original_width, original_height = input_image.size
#             input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
#             input_image = transforms.ToTensor()(input_image).unsqueeze(0)
#
#             # PREDICTION
#             input_image = input_image.to(device)
#             features = encoder(input_image)
#             outputs = depth_decoder(features)
#
#             disp = outputs[("disp", 0)]
#
#             disp_resized = torch.nn.functional.interpolate(
#                 disp, (original_height, original_width), mode="bilinear", align_corners=False)
#
#             # Saving numpy file
#             output_name = os.path.splitext(os.path.basename(image_path))[0]
#             # output_name = os.path.splitext(image_path)[0].split('/')[-1]
#             scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
#
#             name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
#             np.save(name_dest_npy, scaled_disp.cpu().numpy())
#
#             # Saving colormapped depth image
#             disp_resized_np = disp_resized.squeeze().cpu().numpy()
#             vmax = np.percentile(disp_resized_np, 95)
#             normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
#             mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
#             colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
#             im = pil.fromarray(colormapped_im)
#
#             name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
#             im.save(name_dest_im)
#
#             print("   Processed {:d} of {:d} images - saved predictions to:".format(
#                 idx + 1, len(paths)))
#             print("   - {}".format(name_dest_im))
#             print("   - {}".format(name_dest_npy))
#
#
#     print('-> Done!')
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     test_simple(args)


#####-----------------------------摄像头获取----------------------------------------
#
# import os
# import sys
# import glob
# import argparse
# import numpy as np
# import PIL.Image as pil
# import matplotlib as mpl
# import matplotlib.cm as cm
#
# import torch
# from torchvision import transforms
#
# import cv2
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
#
# # 假设networks模块包含了你需要的模型定义
# import networks
# from layers import disp_to_depth
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Real-time depth estimation using a webcam.')
#     parser.add_argument('--load_weights_folder', type=str, help='path of a pretrained model to use', required=True)
#     parser.add_argument('--model', type=str, help='name of a pretrained model to use', default="lite-mono")
#     parser.add_argument("--no_cuda", help='if set, disables CUDA', action='store_true')
#     return parser.parse_args()
#
# def test_simple(args):
#     assert args.load_weights_folder is not None, "You must specify the --load_weights_folder parameter"
#
#     if torch.cuda.is_available() and not args.no_cuda:
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     print("-> Loading model from ", args.load_weights_folder)
#     encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
#     decoder_path = os.path.join(args.load_weights_folder, "depth.pth")
#
#     encoder_dict = torch.load(encoder_path)
#     decoder_dict = torch.load(decoder_path)
#
#     feed_height = encoder_dict['height']
#     feed_width = encoder_dict['width']
#
#     encoder = networks.LiteMono(model=args.model, height=feed_height, width=feed_width)
#     model_dict = encoder.state_dict()
#     encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
#     encoder.to(device)
#     encoder.eval()
#
#     depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
#     depth_model_dict = depth_decoder.state_dict()
#     depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
#     depth_decoder.to(device)
#     depth_decoder.eval()
#
#     cap = cv2.VideoCapture(1)
#     if not cap.isOpened():
#         print("Error: Could not open camera")
#         exit()
#
#     print("-> Starting camera feed")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Can't receive frame (stream end?). Exiting ...")
#             break
#
#         input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_image = pil.fromarray(input_image)
#         input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
#         input_image = transforms.ToTensor()(input_image).unsqueeze(0)
#
#         input_image = input_image.to(device)
#         features = encoder(input_image)
#         outputs = depth_decoder(features)
#
#         disp = outputs[("disp", 0)]
#         disp_resized = torch.nn.functional.interpolate(disp, (frame.shape[0], frame.shape[1]), mode="bilinear", align_corners=False)
#
#         scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
#         disp_resized_np = disp_resized.squeeze().cpu().numpy()
#         vmax = np.percentile(disp_resized_np, 95)
#         normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
#         mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
#         colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
#         im = pil.fromarray(colormapped_im)
#
#         cv2.imshow('Depth Estimation', cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
#
#         if cv2.waitKey(1) == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#     print('-> Done!')
#
# if __name__ == '__main__':
#     args = parse_args()
#     test_simple(args)

# ggggggggg------------------------------------------
# from __future__ import absolute_import, division, print_function
#
# import os
# import sys
# import argparse
# import numpy as np
# import torch
# from torchvision import transforms
# import networks
# from layers import disp_to_depth
# import cv2
#
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Simple testing function for Lite-Mono models on video input.')
#
#     parser.add_argument('--load_weights_folder', type=str,
#                         help='path of a pretrained model to use', required=True)
#
#     parser.add_argument('--model', type=str,
#                         help='name of a pretrained model to use',
#                         default="lite-mono",
#                         choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"])
#
#     parser.add_argument("--no_cuda",
#                         help='if set, disables CUDA',
#                         action='store_true')
#
#     return parser.parse_args()
#
# def test_video(args):
#     """Function to predict depth from a video stream
#     """
#     assert args.load_weights_folder is not None, \
#         "You must specify the --load_weights_folder parameter"
#
#     if torch.cuda.is_available() and not args.no_cuda:
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     print("-> Loading model from ", args.load_weights_folder)
#     encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
#     decoder_path = os.path.join(args.load_weights_folder, "depth.pth")
#
#     encoder_dict = torch.load(encoder_path)
#     decoder_dict = torch.load(decoder_path)
#
#     # extract the height and width of image that this model was trained with
#     feed_height = encoder_dict['height']
#     feed_width = encoder_dict['width']
#
#     # LOADING PRETRAINED MODEL
#     print("   Loading pretrained encoder")
#     encoder = networks.LiteMono(model=args.model,
#                                 height=feed_height,
#                                 width=feed_width)
#
#     model_dict = encoder.state_dict()
#     encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
#
#     encoder.to(device)
#     encoder.eval()
#
#     print("   Loading pretrained decoder")
#     depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
#     depth_model_dict = depth_decoder.state_dict()
#     depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
#
#     depth_decoder.to(device)
#     depth_decoder.eval()
#
#     # 开始摄像头捕获
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open video stream.")
#         return
#
#     print("-> Starting video stream")
#
#     with torch.no_grad():
#         while True:
#             # 读取视频帧
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Failed to capture image.")
#                 break
#
#             # 调整帧的大小以匹配模型的输入尺寸
#             input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             input_image = cv2.resize(input_image, (feed_width, feed_height))
#             input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
#
#             # 深度预测
#             features = encoder(input_image)
#             outputs = depth_decoder(features)
#             disp = outputs[("disp", 0)]
#
#             disp_resized = torch.nn.functional.interpolate(
#                 disp, (frame.shape[0], frame.shape[1]), mode="bilinear", align_corners=False)
#
#             disp_resized_np = disp_resized.squeeze().cpu().numpy()
#             vmax = np.percentile(disp_resized_np, 95)
#             normalized_disp = disp_resized_np / vmax
#
#             # 显示深度图
#             depth_map = (normalized_disp * 255).astype(np.uint8)
#             depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
#
#             combined = np.hstack((frame, depth_map_color))
#             cv2.imshow('Depth Estimation', combined)
#
#             # 按 'q' 键退出
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     args = parse_args()
#     test_video(args)
#
#


#ggggggggggggggggggggggggggggggggggg视频输入
# from __future__ import absolute_import, division, print_function
#
# import os
# import argparse
# import numpy as np
# import torch
# from torchvision import transforms
# import networks
# from layers import disp_to_depth
# import cv2
#
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Simple testing function for Lite-Mono models on video input.')
#
#     parser.add_argument('--video_path', type=str,
#                         help='path to the input video file', required=True)
#
#     parser.add_argument('--load_weights_folder', type=str,
#                         help='path of a pretrained model to use', required=True)
#
#     parser.add_argument('--model', type=str,
#                         help='name of a pretrained model to use',
#                         default="lite-mono",
#                         choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"])
#
#     parser.add_argument("--no_cuda",
#                         help='if set, disables CUDA',
#                         action='store_true')
#
#     return parser.parse_args()
#
# def test_video(args):
#     """Function to predict depth from a video file
#     """
#     assert args.load_weights_folder is not None, \
#         "You must specify the --load_weights_folder parameter"
#
#     if torch.cuda.is_available() and not args.no_cuda:
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     print("-> Loading model from ", args.load_weights_folder)
#     encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
#     decoder_path = os.path.join(args.load_weights_folder, "depth.pth")
#
#     encoder_dict = torch.load(encoder_path)
#     decoder_dict = torch.load(decoder_path)
#
#     # extract the height and width of image that this model was trained with
#     feed_height = encoder_dict['height']
#     feed_width = encoder_dict['width']
#
#     # LOADING PRETRAINED MODEL
#     print("   Loading pretrained encoder")
#     encoder = networks.LiteMono(model=args.model,
#                                 height=feed_height,
#                                 width=feed_width)
#
#     model_dict = encoder.state_dict()
#     encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
#
#     encoder.to(device)
#     encoder.eval()
#
#     print("   Loading pretrained decoder")
#     depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
#     depth_model_dict = depth_decoder.state_dict()
#     depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
#
#     depth_decoder.to(device)
#     depth_decoder.eval()
#
#     # 从视频文件读取
#     cap = cv2.VideoCapture(args.video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {args.video_path}")
#         return
#
#     print("-> Processing video: ", args.video_path)
#
#     with torch.no_grad():
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break  # 视频读取结束
#
#             # 调整帧的大小以匹配模型的输入尺寸
#             input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             input_image = cv2.resize(input_image, (feed_width, feed_height))
#             input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
#
#             # 深度预测
#             features = encoder(input_image)
#             outputs = depth_decoder(features)
#             disp = outputs[("disp", 0)]
#
#             disp_resized = torch.nn.functional.interpolate(
#                 disp, (frame.shape[0], frame.shape[1]), mode="bilinear", align_corners=False)
#
#             disp_resized_np = disp_resized.squeeze().cpu().numpy()
#             vmax = np.percentile(disp_resized_np, 95)
#             normalized_disp = disp_resized_np / vmax
#
#             # 显示深度图
#             depth_map = (normalized_disp * 255).astype(np.uint8)
#             depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
#
#             combined = np.hstack((frame, depth_map_color))
#             cv2.imshow('Depth Estimation', combined)
#
#             # 按 'q' 键退出
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     args = parse_args()
#     test_video(args)

#
# ################################小分辨率
# from __future__ import absolute_import, division, print_function
#
# import os
# import sys
# import glob
# import argparse
# import time
#
# import numpy as np
# import PIL.Image as pil
# import matplotlib as mpl
# import matplotlib.cm as cm
#
# import torch
# from torchvision import transforms, datasets
#
# import networks
# from layers import disp_to_depth
# import cv2
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Simple testing function for Lite-Mono models with video input.')
#
#     parser.add_argument('--video_path', type=str,
#                         help='path to a test video file', required=True)
#
#     parser.add_argument('--load_weights_folder', type=str,
#                         help='path of a pretrained model to use', required=True)
#
#     parser.add_argument('--model', type=str,
#                         help='name of a pretrained model to use',
#                         default="lite-mono",
#                         choices=[
#                             "lite-mono",
#                             "lite-mono-small",
#                             "lite-mono-tiny",
#                             "lite-mono-8m"])
#
#     parser.add_argument("--no_cuda",
#                         help='if set, disables CUDA',
#                         action='store_true')
#
#     return parser.parse_args()
# #
# #
# # def test_video(args):
# #     """Function to predict depth from a video file"""
# #     assert args.load_weights_folder is not None, "You must specify the --load_weights_folder parameter"
# #
# #     if torch.cuda.is_available() and not args.no_cuda:
# #         device = torch.device("cuda")
# #     else:
# #         device = torch.device("cpu")
# #
# #     print("-> Loading model from ", args.load_weights_folder)
# #     encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
# #     decoder_path = os.path.join(args.load_weights_folder, "depth.pth")
# #
# #     encoder_dict = torch.load(encoder_path)
# #     decoder_dict = torch.load(decoder_path)
# #
# #     # extract the height and width of image that this model was trained with
# #     feed_height = encoder_dict['height']
# #     feed_width = encoder_dict['width']
# #
# #     # LOADING PRETRAINED MODEL
# #     print("   Loading pretrained encoder")
# #     encoder = networks.LiteMono(model=args.model,
# #                                 height=feed_height,
# #                                 width=feed_width)
# #
# #     model_dict = encoder.state_dict()
# #     encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
# #
# #     encoder.to(device)
# #     encoder.eval()
# #
# #     print("   Loading pretrained decoder")
# #     depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
# #     depth_model_dict = depth_decoder.state_dict()
# #     depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
# #
# #     depth_decoder.to(device)
# #     depth_decoder.eval()
# #
# #     # 从视频文件读取
# #     cap = cv2.VideoCapture(args.video_path)
# #     if not cap.isOpened():
# #         print(f"Error: Could not open video file {args.video_path}")
# #         return
# #
# #     print("-> Processing video: ", args.video_path)
# #
# #     with torch.no_grad():
# #
# #         while cap.isOpened():
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break  # 视频读取结束
# #
# #             # 调整帧的大小以匹配模型的输入尺寸
# #             input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #             input_image = cv2.resize(input_image, (feed_width, feed_height))
# #             input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
# #             start_time = time.time()
# #
# #             # Run the model inference
# #             features = encoder(input_image)
# #             outputs = depth_decoder(features)
# #
# #             # Calculate inference time
# #             inference_time = time.time() - start_time
# #             print(f"Inference time = {inference_time:.4f} seconds")
# #
# #             # 深度预测
# #             # features = encoder(input_image)
# #             # outputs = depth_decoder(features)
# #             disp = outputs[("disp", 0)]
# #
# #             disp_resized = torch.nn.functional.interpolate(
# #                 disp, (frame.shape[0], frame.shape[1]), mode="bilinear", align_corners=False)
# #
# #             disp_resized_np = disp_resized.squeeze().cpu().numpy()
# #             vmax = np.percentile(disp_resized_np, 95)
# #             normalized_disp = disp_resized_np / vmax
# #
# #             # 显示深度图
# #             depth_map = (normalized_disp * 255).astype(np.uint8)
# #             depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
# #
# #             # 缩小显示尺寸
# #             small_frame = cv2.resize(frame, (640, 192))  # 缩小原视频帧
# #             small_depth_map_color = cv2.resize(depth_map_color, (640, 192))  # 缩小深度图
# #
# #             combined = np.hstack((small_frame, small_depth_map_color))
# #             cv2.imshow('Depth Estimation', combined)
# #
# #             # 按 'q' 键退出
# #             if cv2.waitKey(1) & 0xFF == ord('q'):
# #                 break
# #
# #     cap.release()
# #     cv2.destroyAllWindows()
# def test_video(args):
#     """Function to predict depth from a video file"""
#     assert args.load_weights_folder is not None, "You must specify the --load_weights_folder parameter"
#
#     if torch.cuda.is_available() and not args.no_cuda:
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     print("-> Loading model from ", args.load_weights_folder)
#     encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
#     decoder_path = os.path.join(args.load_weights_folder, "depth.pth")
#
#     encoder_dict = torch.load(encoder_path)
#     decoder_dict = torch.load(decoder_path)
#
#     # extract the height and width of image that this model was trained with
#     feed_height = encoder_dict['height']
#     feed_width = encoder_dict['width']
#
#     # LOADING PRETRAINED MODEL
#     print("   Loading pretrained encoder")
#     encoder = networks.LiteMono(model=args.model,
#                                 height=feed_height,
#                                 width=feed_width)
#
#     model_dict = encoder.state_dict()
#     encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
#
#     encoder.to(device)
#     encoder.eval()
#
#     print("   Loading pretrained decoder")
#     depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
#     depth_model_dict = depth_decoder.state_dict()
#     depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
#
#     depth_decoder.to(device)
#     depth_decoder.eval()
#
#     # 从视频文件读取
#     cap = cv2.VideoCapture(args.video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {args.video_path}")
#         return
#
#     # 获取原视频的帧率
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_duration = 1.0 / fps  # 每帧的时间间隔
#
#     print("-> Processing video: ", args.video_path)
#
#     with torch.no_grad():
#
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break  # 视频读取结束
#
#             # 调整帧的大小以匹配模型的输入尺寸
#             input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             input_image = cv2.resize(input_image, (feed_width, feed_height))
#             input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
#             start_time = time.time()
#
#             # Run the model inference
#             features = encoder(input_image)
#             outputs = depth_decoder(features)
#
#             # Calculate inference time
#             inference_time = time.time() - start_time
#             print(f"Inference time = {inference_time:.4f} seconds")
#
#             # 深度预测
#             disp = outputs[("disp", 0)]
#
#             disp_resized = torch.nn.functional.interpolate(
#                 disp, (frame.shape[0], frame.shape[1]), mode="bilinear", align_corners=False)
#
#             disp_resized_np = disp_resized.squeeze().cpu().numpy()
#             vmax = np.percentile(disp_resized_np, 95)
#             normalized_disp = disp_resized_np / vmax
#
#             # 显示深度图
#             depth_map = (normalized_disp * 255).astype(np.uint8)
#             depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
#
#             # 缩小显示尺寸
#             small_frame = cv2.resize(frame, (1024, 320))  # 缩小原视频帧
#             small_depth_map_color = cv2.resize(depth_map_color, (1024, 320))  # 缩小深度图
#
#             combined = np.hstack((small_frame, small_depth_map_color))
#             cv2.imshow('Depth Estimation', combined)
#
#             # 控制帧率，使播放速度与原视频一致
#             time_elapsed = time.time() - start_time
#             if time_elapsed < frame_duration:
#                 time.sleep(frame_duration - time_elapsed)
#
#             # 按 'q' 键退出
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     test_video(args)




