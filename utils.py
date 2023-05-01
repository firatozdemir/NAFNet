import glob, os
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms

def check_saved_output_image_frames(output_path):
    l_files = glob.glob(os.path.join(output_path, '*.jpg'))
    l_files = [os.path.basename(f) for f in l_files]
    l_files = [f.split('.')[0] for f in l_files]
    l_files = [int(f) for f in l_files]
    l_files.sort()
    if len(l_files) == 0:
        print(f'No previously converted frames found. Starting from frame 1.')
        return 1
    frame_ids = l_files
    if len(frame_ids) != frame_ids[-1]:
        print(f'Warning: frame_ids is not continuous. Starting over again.')
        return 1
    else:
        print(f'Found previously converted frames. Starting from frame {frame_ids[-1]}.')
        return frame_ids[-1]
    
class DataGenerator(data.Dataset):
    def __init__(self, path_in, filelist=None):
        self.path_in = path_in
        self.filelist = filelist
        self.frame_ids = None
        if filelist is None:
            self.read_filelist()
        self.filelist_to_frame_ids()
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            lambda x: x[[2,1,0], ...], #BGR to RGB
            ])
    def read_filelist(self):
        self.filelist = glob.glob(os.path.join(self.path_in, '*.jpg'))
        self.filelist.sort()
        print(f'Found {len(self.filelist)} files in {self.path_in}.')    
    def filelist_to_frame_ids(self):
        self.frame_ids = [int(os.path.basename(f).split('.')[0]) for f in self.filelist]
        self.frame_ids.sort()
    def __len__(self):
        return len(self.frame_ids)
    def __getitem__(self, index):
        fname = self.filelist[index]
        im = cv2.imread(fname)
        im = self.transform(im)
        return im
    
class Mp4Generator(data.Dataset):
    def __init__(self, fname_in, frame_start=0, frame_end=10_000):
        self.fname_in = fname_in
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.images = []
        # self.check_total_len()
        self.len = self.read_vid()
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            lambda x: x[[2,1,0], ...], #BGR to RGB
            ])
    # def check_total_len(self):
    #     print(f'Start checking total length.')
    #     vidcap = cv2.VideoCapture(self.fname_in)
    #     success, _ = vidcap.read()
    #     len_ = 1
    #     while success:
    #         success, _ = vidcap.read()
    #         len_ +=1
    #         if len_ % 1000 == 0:
    #             print(f'current len_: {len_}')
    #     print(f'Read vid. Total of {len_} frames found.')

    def read_vid(self):
        print(f'Starting to read vid file. This may take a while.')
        vidcap = cv2.VideoCapture(self.fname_in)
        success, image = vidcap.read()
        if self.frame_start == 0:
            self.images.append(image)
        len_ = 1
        while success:
            success, image = vidcap.read()
            len_ +=1
            if len_ > self.frame_start and len_ <= self.frame_end:
                self.images.append(image)
            elif len_ > self.frame_end:
                break
        print(f'Read vid frame interval. Total of {len_} frames loaded to RAM.')
        return len_

    def __len__(self):
        return self.frame_end-self.frame_start

    def __getitem__(self, index):
        im = self.images[index]
        im = self.transform(im)
        return im

class Create_Images:
    def __init__(self, path_out, size, frame_id=1):
        self.path_out = path_out
        self.size = size
        self.frame_id = frame_id
    def save_im(self, images):
        for im in images:
            # im = tensor2img(im)
            im = im.cpu().numpy()
            im = im.transpose(1,2,0)
            im[im<0] = 0
            im[im>1] = 1
            # im -= im.min()
            # im /= im.max()
            im = (im * 255).astype('uint8')
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{self.path_out}/{self.frame_id:010d}.jpg', im)
            self.frame_id += 1
        
class Create_Video:
    def __init__(self, fname_out, size, frame_rate=30, freq_log=100):
        self.fname_out = fname_out
        self.size = size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(fname_out, fourcc, frame_rate, self.size, True)
        self.frame_id = 1
        self.freq_log = freq_log

    def check_written_object(self):
        with open(self.fname_out, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
        if file_size == 0:
            print(f"Error: Nothing was written to the file {self.fname_out} after {self.frame_id} frames.")
        else:
            print(f"{file_size} bytes written after {self.frame_id} frames to file {self.fname_out}.")

    def add_frames(self, images, save_frame=False):
        for im in images:
            # im = tensor2img(im)
            im = im.cpu().numpy()
            im = im.transpose(1,2,0)
            im[im<0] = 0
            im[im>1] = 1
            # im -= im.min()
            # im /= im.max()
            im = (im * 255).astype('uint8')
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            self.writer.write(im)
            if self.frame_id % self.freq_log == 0:
                self.check_written_object()
            if save_frame:
                cv2.imwrite(f'{os.path.dirname(self.fname_out)}/{self.frame_id:010d}.jpg', im)
            self.frame_id += 1
        
    def close(self):
        self.writer.release()
        print(f'Video saved to {self.fname_out}.')
        
def export(fname_in, path_out, l_frame=None):
    os.makedirs(path_out, exist_ok=True)
    vidcap = cv2.VideoCapture(fname_in)
    success,image = vidcap.read()
    count = 0
    while success:
        fname_out = os.path.join(path_out, f"{count:010d}.jpg")
        if l_frame is not None:
            if count in l_frame:
                cv2.imwrite(fname_out, image)     
            elif count > max(l_frame):
                break
        else:
            cv2.imwrite(fname_out, image)     
        success,image = vidcap.read()
        count += 1
    print(f'Exported {count} frames from {fname_in} to {path_out}.\nLast filename: {fname_out}.')