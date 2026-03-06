import pandas as pd
import numpy as np
import cv2
import bisect
import math
import json
import threading
import os
import struct
import pickle
import time


import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
_imread_executor_pool = ThreadPoolExecutor(max_workers = 16)

from diffcast.tools.utilsss import nonlinpix_to_linpix



def convert_datetime_to_realtime_filepath(date_time, realtime=True):
    """Convert datetime to the realtime radar image filepath

    Parameters
    ----------
    date_time : datetime.datetime

    Returns
    -------
    ret : str
    """
    
    if realtime:
        ret_realtime = os.path.join("%04d" %date_time.year,
                            "%04d" %date_time.year + "%02d" %date_time.month,
                            "%04d" %date_time.year + "%02d" %date_time.month + "%02d" %date_time.day,
                            'input_raw/HKO-10/radarPNG_converted',
                            'RAD%02d%02d%02d%02d%02d00.png'
                            %(date_time.year - 2000, date_time.month, date_time.day,
                            date_time.hour, date_time.minute))

        ret = os.path.join("/home/swirls/operation/swirls_dev/data/archive/", ret_realtime)
    else:
        ret = os.path.join("%04d" %date_time.year,
                        "%02d" %date_time.month,
                        "%02d" %date_time.day,
                        'RAD%02d%02d%02d%02d%02d00.png'
                        %(date_time.year - 2000, date_time.month, date_time.day,
                          date_time.hour, date_time.minute))
        ret = os.path.join("/mnt/VolB/nowcast_swirlsrg11/HKO-10/radarPNG_converted/", ret)
    # ret = os.path.join('/home/nowcast/HKO-10/radarPNG_converted', ret)
    
    return ret



def cv2_read_img(path, read_storage, grayscale, nonlin2lin):
    if grayscale:
        read_storage[:] = cv2.imread(path, 0)
    else:
        read_storage[:] = cv2.imread(path)

    if nonlin2lin : 
        read_storage[:] = nonlinpix_to_linpix(read_storage[:] / 255.0) * 255.0


def quick_read_frames(path_list, im_w=None, im_h=None, resize=False, frame_size=None, grayscale=True, nonlin2lin=False):
    """Multi-thread Frame Loader

    Parameters
    ----------
    path_list : list
    resize : bool, optional
    frame_size : None or tuple

    Returns
    -------

    """
    img_num = len(path_list)
    
    for i in range(img_num):
        if not os.path.exists(path_list[i]):
            dtc = pd.DataFrame(path_list)
            dtc.to_csv('dtc2.csv')
            print(path_list[i])
            raise IOError
    if im_w is None or im_h is None:
        im_w, im_h = quick_imsize(path_list[0])
    
    read_storage = np.empty((img_num, im_h, im_w), dtype=np.uint8)
    
    if img_num == 1:
        cv2_read_img(path=path_list[0], read_storage=read_storage[0], grayscale=grayscale, nonlin2lin=nonlin2lin)
    else:
        future_objs = []
        for i in range(img_num):
            obj = _imread_executor_pool.submit(cv2_read_img, path_list[i], read_storage[i], grayscale, nonlin2lin)
            future_objs.append(obj)
        wait(future_objs)
    if grayscale:
        read_storage = read_storage.reshape((img_num, 1, im_h, im_w))
    else:
        read_storage = read_storage.transpose((0, 3, 1, 2))
    return read_storage[:, ::-1, ...]



def quick_sample_realtime(base_time, in_len=5, freq='12min', nonlin2lin=True, realtime=True):
    """
    skips a lot of checking, unstable
    """
    in_clip = pd.date_range(start=base_time,
                            periods=in_len,
                            freq='-'+freq)[::-1]

    datetime_clip = in_clip
    paths= []
    for datetime in datetime_clip:
        path = convert_datetime_to_realtime_filepath(datetime, realtime=realtime)
        print(path)
        assert os.path.exists(path)
        paths.append(path)
    all_frame_dat = quick_read_frames(path_list=paths,
                                      im_h=480,
                                      im_w=480,
                                      nonlin2lin=nonlin2lin)
    all_frame_dat = np.expand_dims(all_frame_dat, axis=1)
    return all_frame_dat, [datetime_clip]