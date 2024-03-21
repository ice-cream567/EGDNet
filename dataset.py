import h5py

from torch.utils.data import Dataset
import pdb


class Dataset(Dataset):
    def __init__(self,hdf5_name='./data/datasets/val/000.h5'):
        super(Dataset,self).__init__()
        self.data=h5py.File(hdf5_name , 'r')

    def __len__(self):
        return len(self.data['blurry_frame'])

    def __getitem__(self, idx):
        blurry_frame = self.data['blurry_frame'][idx]
        event_map = self.data['event_map'][idx]
        #sharp_frame = self.data['sharp_frame'][idx]
        item = {
                'blurry_frame': blurry_frame,  # (1, 180, 240)
                'event_map': event_map,  # (26, 180, 240)
                #'sharp_frame': sharp_frame,  # (13, 180, 240)
               }
        return item

if __name__ == '__main__':
    dataset = Dataset()
    item = dataset[0]
    pdb.set_trace()
    print(123)
