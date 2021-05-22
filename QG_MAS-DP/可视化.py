# import numpy as np
import cv2
import os
import glob
import imageio


def highcharts(x):
    # 生成阶梯折线图格式
    x_res = [x[0]]
    for i in range(1, len(x)):
        x_res.append(x_res[-1])
        x_res.append(x[i])
    return x_res


def to_video(path, fps=2, video_name="diagram"):
    """
    :param path: 路径
    :param fps: 帧数
    :param video_name: 文件名
    :return: None
    """
    size = (2000, 2000)
    videowriter = cv2.VideoWriter(f"./video/{video_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)
    imgname_lst = glob.glob(f"./{path}/1d_*.png")
    imgname_lst.sort(key=lambda x: os.path.getctime(x))
    print(imgname_lst)
    for i in imgname_lst:
        img = cv2.imread(i)
        videowriter.write(img)


def to_gif(path, duration=0.2, gif_name = "diagram"):
    """
    :param path: 路径
    :param duration: 每张图片停留秒数
    :param gif_name: 文件名
    :return: None
    """
    frames = []
    imgname_lst = glob.glob(f"./{path}/1d_*.png")
    imgname_lst.sort(key=lambda x: os.path.getctime(x))
    print(imgname_lst)
    for image_name in imgname_lst:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(f"./gif/{gif_name}.gif", frames, 'GIF', duration=duration)