# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import numpy as np


# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options

import utils


def main_red():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    img_path = opt['img_path'].get('input_img')
    output_path = opt['img_path'].get('output_img')
    batch_size = opt['batch_size']
    num_workers = opt['num_workers']
    frame_size= (576, 720)
    frame_rate= 30
    frame_id = utils.check_saved_output_image_frames(output_path)

    ## setup data reader
    dataset = utils.DataGenerator(path_in=img_path,)

    ## load model
    opt['dist'] = False
    model = create_model(opt)

    freq_log = 500
    ## setup video exporter
    im_writer = utils.Create_Images(path_out=output_path, size=frame_size, frame_id=frame_id)

    total_frames = len(dataset)
    
    for i_mb in range(int(np.ceil(total_frames/batch_size))):
        i_start = i_mb * batch_size
        i_end = np.min(((i_mb+1) * batch_size, len(dataset)))
        if i_start < frame_id:
            continue
        x = torch.stack([dataset[i] for i in range(i_start, i_end)], dim=0)
    
        model.feed_data(data={'lq': x}) # put data to gpu mem

        if model.opt['val'].get('grids', False):
            model.grids()

        model.test() #execute forward pass

        if model.opt['val'].get('grids', False):
            model.grids_inverse()

        visuals = model.get_current_visuals()
        # sr_img = tensor2img(visuals['result'])
        im_writer.save_im(visuals['result'])
        # vid_writer.add_frames(visuals['result'], save_frame=False)

        if (i_end) % freq_log <= batch_size:
            print(f'finished exporting {i_end}/{total_frames} of the frames.')


    print(f'finished exporting all {total_frames} frames.')


if __name__ == '__main__':
    main_red()

