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

# class DataLoader():
#     def __init__(self, dataset, batch_size, num_workers):
#         self.dataloader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=int(num_workers),
#             drop_last=True
#         )
#         self.batch_size = batch_size
#     def set_epoch(self, epoch):
#         self.dataset.current_epoch = epoch

#     def load_data(self):
#         return self

#     def __len__(self):
#         """Return the number of data in the dataset"""
#         return len(self.dataset)

#     def __iter__(self):
#         """Return a batch of data"""
#         for i, data in enumerate(self.dataloader):
#             yield data

def main_sidd():
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


    ## load model
    opt['dist'] = False
    model = create_model(opt)

    freq_log = 500
    ## setup video exporter
    # vid_writer = utils.Create_Video(fname_out=output_path, size=frame_size, frame_rate=frame_rate, freq_log=freq_log)
    im_writer = utils.Create_Images(path_out=output_path, size=frame_size, frame_id=frame_id)

    num_frames_at_a_time = 15_000
    total_frames = 85_130
    l_frame_intervals = np.linspace(0, 85130, int(np.ceil(total_frames/num_frames_at_a_time))+1, dtype=int)

    
    for i_fr in range(len(l_frame_intervals)-1): 
        frame_start = l_frame_intervals[i_fr]
        frame_end = l_frame_intervals[i_fr+1]
        print(f'Starting with frame intervals {frame_start} to {frame_end}.')
        ## setup data reader
        dataset = utils.Mp4Generator(fname_in=img_path, frame_start=frame_start, frame_end=frame_end)
        
        for i_mb in range(int(np.ceil(len(dataset)/batch_size))):
            i_start = i_mb * batch_size
            i_end = np.min(((i_mb+1) * batch_size, len(dataset)))
            if frame_start + i_start < frame_id:
                continue
            x = torch.stack([dataset[i] for i in range(i_start, i_end)], dim=0)
        
        # img = img2tensor(img, bgr2rgb=True, float32=True)
        # for i_mb, x in enumerate(dataloader):
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

            if (i_end + frame_start) % freq_log <= batch_size:
                print(f'finished exporting {i_end + frame_start}/{frame_end} of the frames.')


        print(f'finished exporting {frame_end}/{len(l_frame_intervals)} of the frames.')


if __name__ == '__main__':
    main_sidd()

