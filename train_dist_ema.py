import os, sys, argparse, copy, time, pickle, six, math
import PIL
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as datasets
from network.spgnet import SPGNet
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters, reduce_tensor
from torch.cuda.amp import autocast, GradScaler
from randaug import RandAugment
from utils import ModelEma


class ImageFolderLMDB(torch.utils.data.Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transform

    def __getitem__(self, idx):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[idx])
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        label = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.length


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


class PIL2CV2(object):

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        return np.asarray(img)


class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]


def get_args():
    parser = argparse.ArgumentParser("MAD")
    parser.add_argument('--amp', default=True, action='store_true')
    parser.add_argument('--randaug', default=True, action='store_true')
    parser.add_argument('--cv2', default=False, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str, default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--num-epochs', type=int, default=320, help='total epochs')
    parser.add_argument('--lr', type=float, default=0.4, help='init learning rate')
    parser.add_argument('--min-lr', type=float, default=0.0001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--mixup', type=float, default=0.0, help='mixup beta')
    parser.add_argument('--cutmix', type=float, default=0.0, help='cutmix beta')
    parser.add_argument('--cutmix-probability', type=float, default=0.5, help='cutmix beta')
    parser.add_argument('--ema', type=bool, default=True, help='ema usage flag (default: off)')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='')

    parser.add_argument('--auto-continue', type=bool, default=True, help='auto continue')
    parser.add_argument('--seed', type=int, default="613", help='random seed')

    parser.add_argument('--data-dir', type=str, default='../../autodl-tmp/imagenet2021', help='path to dataset')
    parser.add_argument('--dump-dir', type=str, default="./checkpoint")

    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args = parser.parse_args()
    return args

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def adjust_lr(all_iters, args):
    warm_iters = args.warm_iters
    total_iters = args.total_iters

    if all_iters < warm_iters:
        lr = args.min_lr + args.lr * all_iters / warm_iters

    if all_iters >= warm_iters:
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (math.cos(math.pi * (all_iters - warm_iters) / (total_iters - warm_iters)) + 1)

    return lr


def main():
    args = get_args()

    world_size = int(os.environ['WORLD_SIZE'])
    print(f"RANK and WORLD_SIZE in environ: {args.local_rank}/{world_size}")

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:31624', world_size=world_size, rank=args.local_rank)
    torch.distributed.barrier()

    seed = args.seed + args.local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    use_gpu = False
    device = torch.device("cpu")
    if torch.cuda.is_available():
        use_gpu = True
        device = torch.device("cuda")
    
    if args.cv2:
        from cv2_transform import transforms
    else:
        import torchvision.transforms as transforms

    args.train_dir = os.path.join(args.data_dir, "train")
    args.val_dir = os.path.join(args.data_dir, "val")

    assert os.path.exists(args.train_dir)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.02, 0.40), value=0.5),
    ])
    if args.randaug:
        train_transforms.transforms.insert(0, RandAugment(1, 13))
    if args.cv2:
        train_transforms.transforms.insert(0, PIL2CV2())
    if args.local_rank == 0:
        print(train_transforms)

    train_dataset = datasets.ImageFolder(args.train_dir, train_transforms)
    sampler_train = torch.utils.data.DistributedSampler(train_dataset, num_replicas=torch.distributed.get_world_size(), rank=args.local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler_train, num_workers=4, pin_memory=use_gpu, drop_last=True)
    train_dataprovider = DataIterator(train_loader)

    assert os.path.exists(args.val_dir)
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    if args.cv2:
        val_transforms.transforms.insert(0, PIL2CV2())
    val_dataset = datasets.ImageFolder(args.val_dir, val_transforms)
    indices = np.arange(args.local_rank, len(val_dataset), torch.distributed.get_world_size())
    sampler_val = SubsetRandomSampler(indices)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=250, sampler=sampler_val, shuffle=False, num_workers=4, pin_memory=use_gpu, drop_last=False)
    val_dataprovider = DataIterator(val_loader)
    print('load data successfully')

    iters_epoch = 1281167 // (args.batch_size * world_size)
    args.total_iters = iters_epoch * args.num_epochs
    args.warm_iters = iters_epoch * 5
    args.display_interval = iters_epoch
    args.save_interval = iters_epoch * 10
    args.val_interval = iters_epoch * 10
    args.max_accuracy = 0.0
    
    model = SPGNet(version="s2p6")
    # model = SPGNet(version="s2p7")
    # model = SPGNet(version="s2p8")
    # model = SPGNet(version="s2p9")

    model = model.to(device)
    if args.ema:
        model_ema = ModelEma(model, decay=args.ema_decay, device='', resume='')

    optimizer = torch.optim.SGD(get_parameters(model), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion_smooth = CrossEntropyLabelSmooth(1000, args.label_smooth)

    if use_gpu:
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[args.local_rank], output_device=args.local_rank)
        loss_function = criterion_smooth.cuda()
    else:
        loss_function = criterion_smooth

    if args.amp:
        scaler = GradScaler()

    all_iters = 0
    if args.auto_continue:
        lastest_model, iters = get_lastest_model(args.dump_dir)
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location='cpu')
            load_checkpoint(model, checkpoint)
            print('load from checkpoint, {}'.format(lastest_model))
        if args.ema:
            ema_lastest_model, _ = get_lastest_model(args.dump_dir, "ema")
            if ema_lastest_model is not None:
                model_ema._load_checkpoint(ema_lastest_model)
                print('load from ema checkpoint, {}'.format(ema_lastest_model))
        
    args.optimizer = optimizer
    args.loss_function = loss_function
    if args.amp:
        args.scaler = scaler
    args.train_dataprovider = train_dataprovider
    args.val_dataprovider = val_dataprovider

    if args.eval:
        if args.eval_resume is not None:
            checkpoint = torch.load(args.eval_resume, map_location=None if use_gpu else 'cpu')
            load_checkpoint(model, checkpoint)
            validate(model, device, args, all_iters=all_iters)
        exit(0)

    while all_iters < args.total_iters:
        all_iters = train(model, model_ema if args.ema else None, device, args, val_interval=args.val_interval, all_iters=all_iters)
        validate(model, device, args, all_iters=all_iters)
        if args.ema:
            validate(model_ema.ema, device, args, all_iters=all_iters)

    if args.ema:
        model_ema.bn_update(train_loader)
        validate(model_ema.ema, device, args, all_iters=all_iters)
        if args.local_rank == 0:
            save_checkpoint(args.dump_dir, {'state_dict': model.state_dict()}, args.total_iters, tag='bnps-')
            save_checkpoint(args.dump_dir, {'state_dict_ema': model_ema.ema.state_dict()}, args.total_iters, tag='bnps-ema-')
    else:
        validate(model, device, args, all_iters=all_iters)
        if args.local_rank == 0:
            save_checkpoint(args.dump_dir, {'state_dict': model.state_dict()}, args.total_iters, tag='bnps-')


def train(model, model_ema, device, args, val_interval, all_iters=None):

    optimizer = args.optimizer
    if args.amp:
        scaler = args.scaler
    loss_function = args.loss_function
    train_dataprovider = args.train_dataprovider

    t1 = time.time()
    data_time = 0
    Top1, Top5 = 0.0, 0.0
    model.train()
    for iters in range(1, val_interval + 1):
        lr = adjust_lr(all_iters, args)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.defaults['lr'] = lr

        optimizer.zero_grad()
        all_iters += 1
        d_st = time.time()
        data, target = train_dataprovider.next()
        target = target.type(torch.LongTensor)
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        data_time += (time.time() - d_st)

        probability = np.random.rand()
        if (probability < args.cutmix_probability) and (args.cutmix > 0):
            # generate mixed sample
            lam = np.random.beta(args.cutmix, args.cutmix)
            rand_index = torch.randperm(data.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
            data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        elif args.mixup > 0:
            # generate mixed sample
            lam = np.random.beta(args.mixup, args.mixup)
            rand_index = torch.randperm(data.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            data = data * lam + data[rand_index] * (1. - lam)

        if args.amp:
            with autocast():
                if ((probability < args.cutmix_probability) and (args.cutmix > 0)) or args.mixup > 0:
                    output = model(data)
                    loss = loss_function(output, target_a) * lam + loss_function(output, target_b) * (1. - lam)
                elif args.mixup == 0:
                    output = model(data)
                    loss = loss_function(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.set_grad_enabled(True):
                if ((probability < args.cutmix_probability) and (args.cutmix > 0)) or args.mixup > 0:
                    output = model(data)
                    loss = loss_function(output, target_a) * lam + loss_function(output, target_b) * (1. - lam)
                elif args.mixup == 0:
                    output = model(data)
                    loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        Top1 += prec1.item() / 100
        Top5 += prec5.item() / 100

        if all_iters % args.display_interval == 0 and args.local_rank == 0:
            printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, lr, loss.item()) + \
                        'Top-1 = {:.6f},\t'.format(Top1 / args.display_interval) + \
                        'Top-5 = {:.6f},\t'.format(Top5 / args.display_interval) + \
                        'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time / args.display_interval, (time.time() - t1) / args.display_interval)
            print(printInfo)
            t1 = time.time()
            data_time = 0
            Top1, Top5 = 0.0, 0.0

        if all_iters % args.save_interval == 0 and args.local_rank == 0:
            save_checkpoint(args.dump_dir, {'state_dict': model.state_dict()}, all_iters)
            if args.ema:
                save_checkpoint(args.dump_dir, {'state_dict_ema': model_ema.ema.state_dict()}, all_iters, tag='ema-')

    return all_iters

def validate(model, device, args, all_iters=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_dataprovider = args.val_dataprovider

    model.eval()
    max_val_iters = 200 // torch.distributed.get_world_size()
    t1  = time.time()
    with torch.no_grad():
        for _ in range(1, max_val_iters + 1):
            data, target = val_dataprovider.next()
            target = target.type(torch.LongTensor)
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if args.amp:
                with autocast():
                    output = model(data)
                    loss = loss_function(output, target)
            else:
                output = model(data)
                loss = loss_function(output, target)
            
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
            loss = reduce_tensor(loss)

            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    if args.local_rank == 0:
        args.max_accuracy = max(args.max_accuracy, top1.avg / 100)
        logInfo = 'TEST Iter {},{}: loss = {:.6f},\t'.format(all_iters, max_val_iters, objs.avg) + \
                'Top-1 = {:.6f},\t'.format(top1.avg / 100) + \
                'Top-5 = {:.6f},\t'.format(top5.avg / 100) + \
                'Max Acc = {:.6f},\t'.format(args.max_accuracy) + \
                'val_time = {:.6f}'.format(time.time() - t1)
        print(logInfo)

def load_checkpoint(net, checkpoint):
    from collections import OrderedDict

    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    for k in checkpoint:
        k2 = 'module.'+k if not k.startswith('module.') else k
        temp[k2] = checkpoint[k]

    net.load_state_dict(temp, strict=True)

if __name__ == "__main__":
    main()


