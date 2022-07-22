import os
import sys
import time
import json
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
from tqdm import tqdm

sys.path.insert(0, '../../../')
from datasets import GCNDataLayer as DataLayer
from models import GCN as Model
import utils as utl
import config as cfg

def to_device(x, device):
    return x.to(device)  # .transpose(0,1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--inputs', default='camera', type=str)
    parser.add_argument('--cause', default='interactive', type=str)
    parser.add_argument('--gpu', default='0, 1', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--lr', default=5e-03, type=float)  #5e-04
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--time_steps', default=3, type=int)
    parser.add_argument('--partial_conv', default=True, type=bool)  # True
    parser.add_argument('--data_augmentation', default=True, type=bool)
    parser.add_argument('--dist', default=False, type=bool)
    parser.add_argument('--fusion', default='attn',
                        choices=['avg', 'gcn', 'attn'], type=str)

    args = cfg.parse_args(parser)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(args.inputs, args.time_steps,
                  partialConv=args.partial_conv, fusion=args.fusion)

    # url = 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth'
    # model.loadmodel(url)

    # state_dict = torch.load(
    #     "snapshots/interactive/2022-7-13_184135_w_dataAug_attn/inputs-camera-epoch-1.pth")
    # state_dict_copy = {}
    # for key in state_dict.keys():
    #     state_dict_copy[key[7:]] = state_dict[key]
    # model.load_state_dict(state_dict_copy)

    model = nn.DataParallel(model).to(device)
    print("Model Parameters:", count_parameters(model))

    softmax = nn.Softmax(dim=1).to(device)
    weights = [0.20, 1.0]
    # weights = [0.15, 1.0]
    class_weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    camera_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5],
        #                      [0.5, 0.5, 0.5],),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

    ])

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object = datetime.fromtimestamp(timestamp)
    if not args.data_augmentation:
        data_augmentation = 'wo_dataAug'
    else:
        data_augmentation = 'w_dataAug'
    result_name = "{}-{}-{}_{:02d}{:02d}{:02d}_{}_{}_result.json".format(dt_object.year, dt_object.month, dt_object.day,
                                                                         dt_object.hour, dt_object.minute,
                                                                         dt_object.second, data_augmentation, args.fusion)
    formated_time = "{}-{}-{}_{:02d}{:02d}{:02d}".format(dt_object.year, dt_object.month, dt_object.day, dt_object.hour,
                                                         dt_object.minute, dt_object.second)

# print(args)

for epoch in range(1, args.epochs+1):
    data_sets = {
        phase: DataLayer(
            data_root=args.data_root,
            cause=args.cause,
            sessions=getattr(args, phase+'_session_set'),
            camera_transforms=camera_transforms,
            time_steps=args.time_steps,
            data_augmentation=args.data_augmentation,
            dist=args.dist
        )
        for phase in args.phases
    }

    data_loaders = {
        phase: data.DataLoader(
            data_sets[phase],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        for phase in args.phases
    }

    losses = {phase: 0.0 for phase in args.phases}
    vel_metrics = []
    target_metrics = []
    mAP = 0.0

    start = time.time()
    for phase in args.phases:
        training = phase == 'train'
        if training:
            model.train(True)
        else:
            if epoch % args.test_interval == 0 or epoch > 15:
                model.train(False)
            else:
                continue

        with torch.set_grad_enabled(training):
            with tqdm(data_loaders[phase], unit="batch") as tepoch:
                for batch_idx, (camera_inputs, trackers, mask, dist_mask, vel_target) \
                        in enumerate(tepoch):

                    tepoch.set_description(f"Epoch {epoch:2}/{args.epochs}")
                    batch_size = camera_inputs.shape[0]

                    camera_inputs = to_device(
                        camera_inputs, device)  # (bs, t, c , w, h)
                    mask = mask.to(device)
                    dist_mask = dist_mask.to(device)
                    trackers = to_device(trackers, device)  # (bs, t, n, 4)
                    vel_target = to_device(vel_target, device).view(-1)  # (bs)

                    vel = model(camera_inputs, trackers,
                                device, dist_mask, mask)

                    print(np.argmax(vel.detach().to('cpu').numpy(), axis=1))
                    print(np.array(vel_target.detach().to('cpu').numpy()))
                    print(vel)

                    vel_loss = criterion(vel, vel_target)
                    loss = vel_loss
                    # print('batch idx:', batch_idx, '/',
                    #     len(data_loaders[phase]), ':', loss.item())
                    losses[phase] += loss.item()*batch_size

                    if args.debug:
                        print(loss.item())

                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    else:
                        vel = softmax(vel).to('cpu').numpy()
                        vel_target = vel_target.to('cpu').numpy()
                        vel_metrics.extend(vel)
                        target_metrics.extend(vel_target)

                    tepoch.set_postfix(loss=loss.item())
                    # if batch_idx > 40:
                    #     break

    end = time.time()

    if epoch % args.test_interval == 0 or epoch > 15:

        result_path = snapshot_path = os.path.join('./results', args.cause)
        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        epoch_result_name = 'inputs-{}-epoch-{}.json'.format(
            args.inputs, epoch)

        mAP, result = utl.compute_result(args.class_index, vel_metrics, target_metrics,
                                         result_path, epoch_result_name)

        result['Epoch'] = epoch
        with open(os.path.join(result_path, result_name), 'a') as f:
            json.dump(result, f)

        snapshot_path = os.path.join(
            './snapshots', args.cause, formated_time+'_'+data_augmentation+'_'+args.fusion)

        if not os.path.isdir(snapshot_path):
            os.makedirs(snapshot_path)

        snapshot_name = 'inputs-{}-epoch-{}.pth'.format(args.inputs, epoch)
        torch.save(model.state_dict(), os.path.join(
            snapshot_path, snapshot_name))

    print('Epoch {:2} | train loss: {:.5f} | test loss: {:.5f} mAP: {:.5f} | '
          'running time: {:.2f} sec'.format(
              epoch,
              losses['train']/len(data_loaders['train'].dataset),
              losses['test']/len(data_loaders['test'].dataset),
              mAP,
              end-start,))
