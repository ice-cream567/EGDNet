import os
from utils import *
from tqdm import tqdm
import numpy as np
import cv2
import skimage.metrics
from dataset import Dataset
from model import PredNet
import torch.nn as nn
class Test(object):
    def __init__(self, args):
        super(Test, self).__init__()
        self.args = args
        self.cuda = torch.cuda.is_available()
        # create loader
        self.test_loader = self.setup_test_loader()
        # create model
        self.model=PredNet()
        if self.cuda:
            self.model = nn.DataParallel(self.model).cuda()
        self.load_model()


    def test(self):
        self.model.eval()
        metrics = {}
        for metric_name in ['MSE', 'PSNR', 'SSIM']:
            metrics[metric_name] = AverageMeter()
        with torch.no_grad():
            idx = 0
            for i_batch, batch in enumerate(tqdm(self.test_loader)):
                pred = self.model(batch)
                image_pred = np.clip(pred.detach().cpu().numpy(), 0, 1)
                image_gt = batch['sharp_frame'].detach().cpu().numpy()

                for i_example in range(image_pred.shape[0]):
                    save_dir = os.path.join(self.args.save_dir,'output','000')
                    os.makedirs(save_dir, exist_ok=True)

                    for i_time in range(image_pred.shape[1]):
                        save_name = os.path.join(save_dir,'{:06d}_{}.png'.format(idx,i_time))
                        cv2.imwrite(save_name, image_pred[i_example, i_time] * 255)
                        gt = np.uint8(image_gt[i_example, i_time] * 255)
                        pred = np.uint8(image_pred[i_example, i_time] * 255)
                        for metric_name, metric in zip(['MSE', 'PSNR', 'SSIM'],
                                                       [skimage.metrics.normalized_root_mse,
                                                        skimage.metrics.peak_signal_noise_ratio,
                                                        skimage.metrics.structural_similarity]):
                            metrics[metric_name].update(metric(gt, pred))
                    idx += 1

        info = 'Low Resolution:\n' \
               'MSE: {:.3f}\tPSNR: {:.3f}\tSSIM: {:.3f}\n' .format(metrics['MSE'].avg,
                                                                metrics['PSNR'].avg,
                                                                metrics['SSIM'].avg)

        print('Results:')
        print(info)

    def setup_test_loader(self):
        test_set = []
        for i in range(1):
            hdf5_name = os.path.join(self.args.data_dir, 'val', '{:03d}.h5'.format(i))
            dataset = Dataset(hdf5_name)
            test_set.append(dataset)
        test_set = torch.utils.data.ConcatDataset(test_set)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                   batch_size=self.args.batch_size,
                                                   num_workers=0,
                                                   shuffle=False,
                                                   )
        return test_loader

    def load_model(self):
        path = os.path.join(self.args.load_dir, 'model.pth')
        self.model.load_state_dict(torch.load(path))
        print('Successfully loaded model from {}'.format(self.args.load_dir))


