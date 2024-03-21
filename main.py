from utils import *
from option import args
import os
def main():
    if args.test_only:
        from test_realdata import Test
        t = Test(args)
        t.test()
    else:
        from train import Train
        t = Train(args)
        for epoch in range(0, args.epoch):
            t.train(epoch)
            if (epoch+1) % 5 == 0:
                t.save_model(epoch)

if __name__=='__main__':
    initialize(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('batch_size={},num_works={}'.format(args.batch_size,args.num_workers))
    main()

