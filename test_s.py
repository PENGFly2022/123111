from __future__ import absolute_import, division, print_function

# # from __future__ import absolute_import, division, print_function
# #
# # import os
# # import sys
# # import glob
# # import argparse
# # import time
# #
# # import numpy as np
# # import PIL.Image as pil
# # import matplotlib as mpl
# # import matplotlib.cm as cm
# #
# # import torch
# # from torchvision import transforms, datasets
# #
# # import networks
# # from layers import disp_to_depth
# # import cv2
# #
# #
# # def parse_args():
# #     parser = argparse.ArgumentParser(
# #         description='Simple testing function for Lite-Mono models with video input.')
# #
# #     parser.add_argument('--video_path', type=str,
# #                         help='path to a test video file', required=True)
# #
# #     parser.add_argument('--load_weights_folder', type=str,
# #                         help='path of a pretrained model to use', required=True)
# #
# #     parser.add_argument('--model', type=str,
# #                         help='name of a pretrained model to use',
# #                         default="lite-mono",
# #                         choices=[
# #                             "lite-mono",
# #                             "lite-mono-small",
# #                             "lite-mono-tiny",
# #                             "lite-mono-8m"])
# #
# #     parser.add_argument('--batch_size', type=int, default=1,
# #                         help='Batch size for video frames during inference')
# #
# #     parser.add_argument("--no_cuda",
# #                         help='if set, disables CUDA',
# #                         action='store_true')
# #
# #     return parser.parse_args()
# #
# #
# # def test_video(args):
# #     """Function to predict depth from a video file with batch processing"""
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
# #     batch_size = args.batch_size
# #     batch_frames = []
# #
# #     with torch.no_grad():
# #         while cap.isOpened():
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break  # 视频读取结束
# #
# #             # 调整帧的大小以匹配模型的输入尺寸
# #             input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #             input_image = cv2.resize(input_image, (feed_width, feed_height))
# #             input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
# #
# #             batch_frames.append(input_image)
# #
# #             # 如果批处理的帧数达到了batch size，开始推理
# #             if len(batch_frames) == batch_size:
# #                 batch_tensor = torch.cat(batch_frames, dim=0)
# #                 start_time = time.time()
# #
# #                 # Run the model inference
# #                 features = encoder(batch_tensor)
# #                 outputs = depth_decoder(features)
# #
# #                 # Calculate inference time
# #                 inference_time = time.time() - start_time
# #                 print(f"Inference time for batch of {batch_size} frames = {inference_time:.4f} seconds")
# #
# #                 # 对每一帧进行后处理
# #                 for i in range(batch_size):
# #                     disp = outputs[("disp", 0)][i:i+1]
# #
# #                     disp_resized = torch.nn.functional.interpolate(
# #                         disp, (frame.shape[0], frame.shape[1]), mode="bilinear", align_corners=False)
# #
# #                     disp_resized_np = disp_resized.squeeze().cpu().numpy()
# #                     vmax = np.percentile(disp_resized_np, 95)
# #                     normalized_disp = disp_resized_np / vmax
# #
# #                     # 显示深度图
# #                     depth_map = (normalized_disp * 255).astype(np.uint8)
# #                     depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
# #
# #                     # 缩小显示尺寸
# #                     small_frame = cv2.resize(frame, (640, 192))  # 缩小原视频帧
# #                     small_depth_map_color = cv2.resize(depth_map_color, (640, 192))  # 缩小深度图
# #
# #                     combined = np.hstack((small_frame, small_depth_map_color))
# #                     cv2.imshow('Depth Estimation', combined)
# #
# #                     # 按 'q' 键退出
# #                     if cv2.waitKey(1) & 0xFF == ord('q'):
# #                         break
# #
# #                 # 清空批次
# #                 batch_frames = []
# #
# #     cap.release()
# #     cv2.destroyAllWindows()
# #
# #
# # if __name__ == '__main__':
# #     args = parse_args()
# #     test_video(args)
#单图片
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
#             name_dest_im = os.path.join(output_directory, "{}lite_disp.jpeg".format(output_name))
#             im.save(name_dest_im)
#
#             print("   Processed {:d} of {:d} images - saved predictions to:".format(
#                 idx + 1, len(paths)))
#             # print("   - {}".format(name_dest_im))
#             print("   - {}".format(name_dest_npy))
#
#
#     print('-> Done!')
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     test_simple(args)
from __future__ import absolute_import, division, print_function

'''
-----------------------------文件夹预测
'''

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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)

    parser.add_argument('--load_weights_folder', type=str,
                        help='path of a pretrained model to use', required=True)

    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        default="lite-mono-8m",
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
    if os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path  # 设置为输入文件夹作为输出文件夹
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

            name_dest_im = os.path.join(output_directory, "{}_8disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))


    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)

#
#

'''
·计算flops  
'''

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
# import torch.utils.data as data
#
# import networks
# from layers import disp_to_depth
# import cv2
# import time
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
#
# # 导入附加库
# from thop import profile, clever_format
#
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Lite-Mono模型的简单测试函数。')
#
#     parser.add_argument('--image_path', type=str,
#                         help='测试图像的路径或图像文件夹路径', required=True)
#
#     parser.add_argument('--load_weights_folder', type=str,
#                         help='要使用的预训练模型的路径',
#                         )
#
#     parser.add_argument('--test',
#                         action='store_true',
#                         help='如果设置，从.txt文件中读取图像',
#                         )
#
#     parser.add_argument('--model', type=str,
#                         help='要使用的预训练模型的名称',
#                         default="lite-mono-8m",
#                         choices=[
#                             "lite-mono",
#                             "lite-mono-small",
#                             "lite-mono-tiny",
#                             "lite-mono-8m"])
#
#     parser.add_argument('--ext', type=str,
#                         help='在文件夹中搜索的图像扩展名', default="png")
#     parser.add_argument("--no_cuda",
#                         help='如果设置，禁用CUDA',
#                         action='store_true')
#
#     return parser.parse_args()
#
# # 自定义数据集类
# class CustomImageDataset(data.Dataset):
#     def __init__(self, image_paths, transform=None):
#         self.image_paths = image_paths
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = pil.open(image_path).convert('RGB')
#         original_width, original_height = image.size
#         if self.transform:
#             image = self.transform(image)
#         return image, image_path, (original_width, original_height)
#
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())
#
# # 定义用于计算 FLOPs 的函数
# def profile_once(encoder, decoder, x):
#     # 对输入数据进行深拷贝，避免影响原始数据
#     x_e = x.clone()
#     x_e = x_e.to(next(encoder.parameters()).device)
#     x_d = encoder(x_e)
#     flops_e, params_e = profile(encoder, inputs=(x_e,), verbose=False)
#     flops_d, params_d = profile(decoder, inputs=(x_d,), verbose=False)
#
#     flops_total, params_total = flops_e + flops_d, params_e + params_d
#
#     # 格式化 FLOPs 和参数数量
#     flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
#     flops_d, params_d = clever_format([flops_d, params_d], "%.3f")
#     flops_total, params_total = clever_format([flops_total, params_total], "%.3f")
#
#     return flops_total, params_total, flops_e, params_e, flops_d, params_d
#
# def test_simple(args):
#     """用于对单个图像或图像文件夹进行预测的函数
#     """
#     assert args.load_weights_folder is not None, \
#         "您必须指定--load_weights_folder参数"
#
#     if torch.cuda.is_available() and not args.no_cuda:
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     print("-> 从以下路径加载模型 ", args.load_weights_folder)
#     encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
#     decoder_path = os.path.join(args.load_weights_folder, "depth.pth")
#
#     encoder_dict = torch.load(encoder_path, map_location=device)
#     decoder_dict = torch.load(decoder_path, map_location=device)
#
#     # 获取该模型训练时使用的图像的高度和宽度
#     feed_height = encoder_dict['height']
#     feed_width = encoder_dict['width']
#
#     # 加载预训练模型
#     print("   加载预训练编码器")
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
#     print("   加载预训练解码器")
#     depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
#     depth_model_dict = depth_decoder.state_dict()
#     depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
#
#     depth_decoder.to(device)
#     depth_decoder.eval()
#
#     # 计算参数数量
#     encoder_params = count_parameters(encoder)
#     decoder_params = count_parameters(depth_decoder)
#     total_params = encoder_params + decoder_params
#
#     print(f"编码器参数总数：{encoder_params:,}")
#     print(f"解码器参数总数：{decoder_params:,}")
#     print(f"模型参数总数：{total_params:,}")
#
#     # 查找输入图像
#     if os.path.isfile(args.image_path) and not args.test:
#         # 仅对单个图像进行测试
#         paths = [args.image_path]
#         output_directory = os.path.dirname(args.image_path)
#     elif os.path.isdir(args.image_path):
#         # 在文件夹中搜索图像
#         paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
#         output_directory = args.image_path
#     else:
#         raise Exception("无法找到 args.image_path: {}".format(args.image_path))
#
#     print("-> 预测 {:d} 张测试图像".format(len(paths)))
#
#     # 创建数据集和DataLoader
#     transform = transforms.Compose([
#         transforms.Resize((feed_height, feed_width)),
#         transforms.ToTensor(),
#     ])
#
#     dataset = CustomImageDataset(paths, transform=transform)
#
#     dataloader = data.DataLoader(
#         dataset,
#         batch_size=16,  # 设置批量大小为16
#         shuffle=False,
#         num_workers=4,  # 根据您的CPU核心数调整
#         pin_memory=True
#     )
#
#     # 准备一个用于计算 FLOPs 的输入张量
#     dummy_input = torch.randn(1, 3, feed_height, feed_width).to(device)
#
#     # 计算编码器和解码器的 FLOPs
#     flops_total, params_total, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder, dummy_input)
#
#     print(f"\n编码器 FLOPs: {flops_e}, 参数: {params_e}")
#     print(f"解码器 FLOPs: {flops_d}, 参数: {params_d}")
#     print(f"模型总 FLOPs: {flops_total}, 总参数: {params_total}\n")
#
#     # 测量推理速度
#     # 预热GPU
#     dummy_input_speed = torch.randn(16, 3, feed_height, feed_width).to(device)
#     for _ in range(10):
#         with torch.no_grad():
#             features = encoder(dummy_input_speed)
#             outputs = depth_decoder(features)
#
#     # 测量推理时间
#     iterations = 100
#     start_time = time.time()
#
#     for _ in range(iterations):
#         with torch.no_grad():
#             features = encoder(dummy_input_speed)
#             outputs = depth_decoder(features)
#
#     end_time = time.time()
#     total_time = end_time - start_time
#     average_time_per_batch = total_time / iterations
#     average_time_per_image = average_time_per_batch / 16  # 批量大小为16
#
#     print(f"{iterations}次迭代的总推理时间：{total_time:.4f} 秒")
#     print(f"每个批次的平均时间：{average_time_per_batch:.4f} 秒")
#     print(f"每张图像的平均时间：{average_time_per_image:.4f} 秒\n")
#
#     # 对每个批次的图像进行预测
#     with torch.no_grad():
#         for idx, (input_images, image_paths, original_sizes) in enumerate(dataloader):
#             input_images = input_images.to(device)
#
#             # 预测
#             features = encoder(input_images)
#             outputs = depth_decoder(features)
#
#             disp = outputs[("disp", 0)]
#
#             # 处理并保存输出
#             for i in range(input_images.size(0)):
#                 # 获取原始尺寸
#                 original_size = original_sizes[i]
#                 original_width, original_height = original_size
#
#                 disp_resized = torch.nn.functional.interpolate(
#                     disp[i:i+1], (original_height, original_width), mode="bilinear", align_corners=False)
#
#                 disp_resized_np = disp_resized.squeeze().cpu().numpy()
#                 vmax = np.percentile(disp_resized_np, 95)
#                 normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
#                 mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
#                 colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
#                 im = pil.fromarray(colormapped_im)
#
#                 output_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
#
#                 name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
#                 im.save(name_dest_im)
#
#                 # 保存numpy文件（如果需要）
#                 scaled_disp, depth = disp_to_depth(disp[i:i+1], 0.1, 100)
#                 name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
#                 np.save(name_dest_npy, scaled_disp.cpu().numpy())
#
#             print("已处理批次 {:d}/{:d}".format(idx + 1, len(dataloader)))
#
#     print('\n-> 完成！')
#
# if __name__ == '__main__':
#     args = parse_args()
#     test_simple(args)
