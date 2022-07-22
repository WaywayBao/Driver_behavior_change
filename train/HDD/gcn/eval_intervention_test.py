
import os
import os.path as osp
import sys
import cv2
import json
import argparse
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms

sys.path.insert(0, '../../../')
import config as cfg
from models import GCN as Model

def vis_test(img, center_x=None, center_y=None, w=None, h=None):
    # width = 1280.0
    # height = 720.0

    width, height = 1, 1

    if w != None:
        print(center_x, center_y, w, h)

        gt_x1 = (center_x-0.5*w)*width
        gt_x2 = (center_x+0.5*w)*width
        gt_y1 = (center_y-0.5*h)*height
        gt_y2 = (center_y+0.5*h)*height

        cv2.rectangle(img, (int(gt_x1), int(gt_y1)),
                      (int(gt_x2), int(gt_y2)), (0, 0, 255), 8)

    cv2.imshow('filename.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def to_device(x, device):
    return x.unsqueeze(0).to(device)


def read_testdata():
    test_set = []
    data_root = '/media/waywaybao_cs10/Disk_2/Retrieve_tool/data_collection/interactive'

    for basic_scene in os.listdir(data_root):
        basic_scene_path = osp.join(data_root, basic_scene, 'variant_scenario')

        for var_scene in os.listdir(basic_scene_path):
            var_scene_path = osp.join(basic_scene_path, var_scene)

            if basic_scene[:2] == '10':
                test_set.append(var_scene_path)

    return test_set


def get_current_time():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dtime = datetime.fromtimestamp(timestamp)
    return dtime.year, dtime.month, dtime.day, dtime.hour, dtime.minute, dtime.second


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default='1', type=str)
    parser.add_argument('--inputs', default='camera', type=str)
    parser.add_argument('--cause', default='interactive', type=str)
    parser.add_argument('--model',
                        default='snapshots/interactive/2022-7-20_214743_w_dataAug_attn/inputs-camera-epoch-20.pth',
                        type=str)
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--partial_conv', default=True, type=bool)
    parser.add_argument('--fusion', default='attn',
                        choices=['avg', 'gcn', 'attn'], type=str)
    parser.add_argument('--vis', action='store_true', default=False)

    args = cfg.parse_args(parser)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(args.inputs, partialConv=args.partial_conv,
                  fusion=args.fusion).to(device)

    state_dict = torch.load(args.model)
    state_dict_copy = {}

    for key in state_dict.keys():
        state_dict_copy[key[7:]] = state_dict[key]

    model.load_state_dict(state_dict_copy)

    model.train(False)
    softmax = nn.Softmax(dim=1).to(device)

    camera_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5],
        #                      [0.5, 0.5, 0.5],),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    time_steps = 3
    time_sample = 1  # 10
    visualize = args.vis

    if visualize:
        year, month, day, hour, minute, second = get_current_time()
        formated_time = f'{year}-{month}-{day}_{hour:02d}{minute:02d}{second:02d}'
        vis_save_path = f'./vis/{args.cause}/{formated_time}'
        if not os.path.isdir(vis_save_path):
            os.makedirs(vis_save_path)

        with open(f"RA/{formated_time}.json", "w") as f:
            json.dump({}, f, indent=4)

    def plot_vel(pred, target, plot_name):
        t = len(pred)
        timestamp = range(1, t+1)
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(timestamp, pred, marker='o', label="Prediction")
        ax.plot(timestamp, target, marker='o', label="Target")
        # Place a legend to the right of this smaller subplot.
        ax.legend(loc='upper right')
        fig.savefig(plot_name)  # save the figure to file
        plt.close(fig)

    def normalize_box(trackers, width, height):
        normalized_trackers = trackers.copy()

        normalized_trackers[:, :, 3] = normalized_trackers[:,
                                                           :, 1] + normalized_trackers[:, :, 3]
        normalized_trackers[:, :, 2] = normalized_trackers[:,
                                                           :, 0] + normalized_trackers[:, :, 2]

        tmp = normalized_trackers[:, :, 0] / width
        normalized_trackers[:, :, 0] = normalized_trackers[:, :, 1] / height
        normalized_trackers[:, :, 1] = tmp
        tmp = trackers[:, :, 2] / width
        normalized_trackers[:, :, 2] = normalized_trackers[:, :, 3] / height
        normalized_trackers[:, :, 3] = tmp

        return normalized_trackers

    def find_tracker(tracking, start, end):

        width = 1280
        height = 720

        # t_array saves timestamps

        t_array = tracking[:, 0]
        tracking_index = tracking[np.where(t_array == end)[0], 1]
        num_object = len(tracking_index)

        trackers = np.zeros(
            [int((end-start)/time_sample+1), num_object+1, 4])  # Tx(N+1)x4
        trackers[:, 0, :] = np.array(
            [0.0, 0.0, width, height])  # Ego bounding box

        for t in range(start, end+1, time_sample):
            current_tracking = tracking[np.where(t_array == t)[0]]
            for i, object_id in enumerate(tracking_index):

                if object_id in current_tracking[:, 1]:
                    bbox = current_tracking[np.where(
                        current_tracking[:, 1] == object_id)[0], 2:6]
                    bbox[:, 0] = np.clip(bbox[:, 0], 0, 1279)
                    bbox[:, 2] = np.clip(bbox[:, 0]+bbox[:, 2], 0, 1279)
                    bbox[:, 1] = np.clip(bbox[:, 1], 0, 719)
                    bbox[:, 3] = np.clip(bbox[:, 1]+bbox[:, 3], 0, 719)
                    trackers[int((t-start)/time_sample), i+1, :] = bbox

        trackers.astype(np.int32)
        normalized_trackers = normalize_box(trackers, width, height)

        return trackers, normalized_trackers, tracking_index

    def visualize_result(test_sample, frame_id, tracker, filename, gt):
        """
            gt: center_x, center_y, w, h
        """
        # width = 1280.0
        # height = 720.0
        width, height = 1, 1

        camera_name = f'{frame_id:08}.png'
        camera_path = osp.join(test_sample, 'rgb/front', camera_name)
        frame = cv2.imread(camera_path)
        box = tracker[-1]  # x1,y1,x2,y2

        gt_x1 = (gt[0]-0.5*gt[2])*width
        gt_x2 = (gt[0]+0.5*gt[2])*width
        gt_y1 = (gt[1]-0.5*gt[3])*height
        gt_y2 = (gt[1]+0.5*gt[3])*height

        cv2.rectangle(frame, (int(gt_x1), int(gt_y1)),
                      (int(gt_x2), int(gt_y2)), (0, 0, 255), 8)

        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(
            box[2]), int(box[3])), (0, 255, 0), 3)

        cv2.imwrite(f'{filename}.png', frame)

        # cv2.imshow('filename.png', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_all_score(test_sample, frame_id, action_logits, trackers, filename, scenario_id_weather, tracking_id, confidence_go):
        # width = 1280.0
        # height = 720.0
        width, height = 1, 1
        colors_BGR = [(220, 20, 60), (240, 128, 128), (255, 182, 193), (0, 255, 0), (127, 255, 212), (103, 255, 255),
                      (0, 255, 255), (0, 191, 255), (30, 144, 255), (0, 0, 153), (204, 51, 153), (255, 0, 255), (0, 0, 255)]
        colors_RGB = [c[::-1] for c in colors_BGR]

        camera_name = str(frame_id).zfill(8)+'.png'
        camera_path = osp.join(test_sample, 'rgb/front', camera_name)
        frame = cv2.imread(camera_path)
        risk_score = dict()

        for idx, score in enumerate(action_logits[1:]):
            color = colors_BGR[idx % len(colors_BGR)]

            risk_score[str(tracking_id[idx])] = np.float(score[0])
            box = trackers[-1][idx+1]  # x1,y1,x2,y2
            cv2.rectangle(frame, (int(box[0]), int(
                box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, f'{score[0]:.4}', (int(box[0]), int(
                box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # {'10_t1-7_1_p_c_r_1_0' : {58:  {100: 50, 103: 60, 101:20}}, \
        #   {59: {, },  60:{,},  ...}, '10_i-1_1_p_c_l_1_j' : {58:{,}}, ...}
        with open(f"RA/{formated_time}.json") as f:
            vision_RA = json.load(f)
        vision_RA.setdefault(str(scenario_id_weather), {})
        vision_RA[str(scenario_id_weather)][frame_id] = risk_score

        with open(f"RA/{formated_time}.json", "w") as f:
            json.dump(vision_RA, f, indent=4)

        cv2.imwrite(filename+'_all.png', frame)
        # cv2.imshow('filename.png', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        object_id = tracking_id
        object_score = np.array(np.array(action_logits)[1:, 0])
        x = np.arange(len(object_id))

        plt.clf()
        plt.bar(x, object_score, color=np.array(
            colors_RGB[:len(object_id)])/255.)
        plt.plot(x, [confidence_go]*len(x), color='black',
                 label='confidence_go', linewidth=0.7)
        plt.xticks(x, object_id)
        plt.xlabel('Object id')
        plt.ylim(0.0, 1.0)
        plt.title('Risk score')
        plt.legend()
        plt.savefig(filename+'_bar.png')
        # plt.show()

    all_test = read_testdata()
    correct = 0
    result_dict = {}

    for cnt, test_sample in enumerate(all_test, 1):
        with torch.set_grad_enabled(False):

            # session = None
            folder = test_sample
            scenario_id = test_sample.split('/')[-3]

            dyn_desc = open(osp.join(folder, 'dynamic_description.json'))
            data = json.load(dyn_desc)
            dyn_desc.close()

            for key in data.keys():
                if key.isdigit():
                    gt_cause_id = data[key]
                    break

            print("===================================================")
            print(test_sample.split('/')[-3], test_sample.split('/')[-1])
            print(gt_cause_id)

            # read testing data (behavior change frame)
            behavior_change_path = test_sample.split('variant_scenario')[
                0]+'behavior_annotation.txt'
            behavior_change_file = open(behavior_change_path, 'r')
            temp = behavior_change_file.readlines()
            frame_id = (int(temp[1].strip()) + int(temp[0].strip()))//2

            # read bbox of risk object
            bbox_path = osp.join(test_sample, 'bbox/front')
            frame_name = f'{frame_id:08}.json'
            json_file = open(osp.join(bbox_path, frame_name))
            data = json.load(json_file)
            json_file.close()

            for actor in data:
                if actor['actor_id'] == gt_cause_id:
                    bbox = actor['box']
                    break

            center_x = float((bbox[0]+bbox[2])/2)     # width
            center_y = float((bbox[1]+bbox[3])/2)     # height
            w = float(bbox[2]-bbox[0])
            h = float(bbox[3]-bbox[1])

            tracking_results = []
            tracking_results = np.load(osp.join(test_sample, 'tracking.npy'))

            start_time = 0  # 59
            use_mask = True

            pred_metrics = []
            target_metrics = []

            et = int(frame_id)
            st = et - (time_steps-1)*time_sample

            trackers, normalized_trackers, tracking_id = find_tracker(
                tracking_results, st, et)
            normalized_trackers = torch.from_numpy(
                normalized_trackers.astype(np.float32)).to(device)
            normalized_trackers = normalized_trackers.unsqueeze(0)
            num_box = len(trackers[0])

            camera_inputs = []
            action_logits = []

            hx = torch.zeros((num_box, 512)).to(device)
            cx = torch.zeros((num_box, 512)).to(device)

            # without intervention
            for l in range(st, et, time_sample):

                # camera_name = 'output{}.png'.format(str(l-1 + start_time))
                camera_name = str(l-1 + start_time).zfill(8)+'.png'
                camera_path = osp.join(test_sample, 'rgb/front', camera_name)

                # save for later usage in intervention
                read_image = Image.open(camera_path).convert('RGB')
                camera_inputs.append(read_image)

                camera_input = camera_transforms(
                    read_image)
                camera_input = np.array(camera_input)

                #########################
                # vis_test(np.array(read_image), center_x, center_y, w, h)
                #########################

                camera_input = to_device(torch.from_numpy(
                    camera_input.astype(np.float32)), device)
                mask = torch.ones((1, 3, 299, 299)).to(device)

                # assign index for RoIAlign
                # box_ind : (BxN)
                box_ind = np.array(
                    [np.arange(1)] * num_box).transpose(1, 0).reshape(-1)
                box_ind = torch.from_numpy(box_ind.astype(np.int32)).to(device)

                if args.partial_conv:
                    camera_input = model.backbone.features(camera_input, mask)
                else:
                    camera_input = model.backbone.features(camera_input)

                # camera_input: 1xCxHxW
                # normalized_trackers: 1xNx4
                # ROIAlign : BxNx1280
                tracker = normalized_trackers[:,
                                              (l - st)//time_sample].contiguous()
                box_ind = box_ind.view(1, num_box)
                feature_input = model.cropFeature(
                    camera_input, tracker, box_ind)
                # check feature_input
                feature_input = feature_input.view(-1, 1280)

                hx, cx = model.step(feature_input, hx, cx)

            updated_feature, _ = model.message_passing(
                hx, normalized_trackers)  # BxH
            vel = model.vel_classifier(model.drop(updated_feature))
            confidence_go = softmax(vel).to('cpu').numpy()[0][0]

            # with intervention
            for i in range(num_box):
                tracker = trackers[:, i, :]
                if i == 0:
                    action_logits.append([0.0, 1.0])
                    continue
                hx = torch.zeros((num_box, 512)).to(device)
                cx = torch.zeros((num_box, 512)).to(device)

                #  trackers: Tx(N+1)x4 (x1, y1, w, h ) without normalization
                #  normalized_trackers: : Tx(N+1)x4 (y1, x1, y2, x2 ) with normalization
                trackers, normalized_trackers, tracking_id = find_tracker(
                    tracking_results, st, et)

                normalized_trackers = torch.from_numpy(
                    normalized_trackers.astype(np.float32)).to(device)
                normalized_trackers = normalized_trackers.unsqueeze(0)

                for l in range(st, et, time_sample):
                    camera_input = np.array(
                        camera_inputs[(l - st)//time_sample])
                    camera_input[int(trackers[(l - st)//time_sample, i, 1]):int(trackers[(l - st)//time_sample, i, 3]),
                                 int(trackers[(l - st)//time_sample, i, 0]):int(trackers[(l - st)//time_sample, i, 2]), :] = 0
                    camera_input = Image.fromarray(np.uint8(camera_input))
                    np_camera_input = np.array(camera_input)

                    camera_input = camera_transforms(camera_input)
                    camera_input = np.array(camera_input)

                    camera_input = to_device(torch.from_numpy(
                        camera_input.astype(np.float32)), device)

                    # assign index for RoIAlign
                    # box_ind : (BxN)
                    box_ind = np.array(
                        [np.arange(1)] * num_box).transpose(1, 0).reshape(-1)
                    box_ind = torch.from_numpy(
                        box_ind.astype(np.int32)).to(device)

                    if not use_mask:
                        mask = torch.ones((1, 3, 299, 299)).to(device)

                    else:
                        mask = np.ones((1, 3, 299, 299))
                        x1 = int(
                            trackers[(l - st) // time_sample, i, 1]/1280*299)  # x1
                        x2 = int(
                            trackers[(l - st) // time_sample, i, 3]/1280*299)  # x2
                        y1 = int(
                            trackers[(l - st) // time_sample, i, 0]/720*299)  # y1
                        y2 = int(
                            trackers[(l - st) // time_sample, i, 2]/720*299)  # y2
                        mask[:, :, x1:x2, y1:y2] = 0
                        mask = torch.from_numpy(
                            mask.astype(np.float32)).to(device)

                    if args.partial_conv:
                        camera_input = model.backbone.features(
                            camera_input, mask)
                    else:

                        camera_input = model.backbone.features(camera_input)

                    tracker = normalized_trackers[:,
                                                  (l - st)//time_sample].contiguous()
                    tracker[:, i, :] = 0
                    box_ind = box_ind.view(1, num_box)
                    feature_input = model.cropFeature(
                        camera_input, tracker, box_ind)

                    # check feature_input
                    feature_input = feature_input.view(-1, 1280)
                    hx, cx = model.step(feature_input, hx, cx)

                intervened_trackers = torch.ones(
                    (1, time_steps, num_box, 4)).to(device)
                intervened_trackers[:, :, i, :] = 0.0

                intervened_trackers = intervened_trackers * normalized_trackers

                updated_feature, _ = model.message_passing(
                    hx, intervened_trackers)  # BxH
                vel = model.vel_classifier(model.drop(updated_feature))
                action_logits.append(softmax(vel).to('cpu').numpy()[0])  # Nx2
                # print(session, start, end, i, trackers[:,i ], action_logits[i])

            if action_logits:
                cause_object_id = np.argmax(np.array(action_logits)[:, 0])
                action_logits = action_logits - confidence_go
                action_logits = np.clip(action_logits, 0, 1)
                print(f"Scenario s_go: {confidence_go}")

                scenario_id_weather = scenario_id + \
                    '_' + test_sample.split('/')[-1]

                if visualize:
                    filename = vis_save_path + '/' + \
                        scenario_id_weather + str(frame_id)
                    visualize_result(test_sample, frame_id, trackers[:, cause_object_id], filename, [
                        center_x, center_y, w, h])
                    draw_all_score(test_sample, frame_id, action_logits, trackers,
                                   filename, scenario_id_weather, tracking_id, confidence_go)

                result_dict[scenario_id_weather + '/' +
                            str(frame_id)] = list(trackers[-1, cause_object_id])

                print(f'Sample: {cnt}/{len(all_test)}')

                if tracking_id[cause_object_id-1] == gt_cause_id:
                    correct += 1
                print(f'Precision: {(correct/cnt)*100:.2f}%')

    print(f'Precision: {(correct/cnt)*100:.2f}%')


json_file_path = '/home/waywaybao_cs10/Desktop/risk_object_identification-master/train/HDD/gcn/testing_result/'+args.cause
if not os.path.isdir(json_file_path):
    os.makedirs(json_file_path)

json_file_path = json_file_path+'/ours.json'
with open(json_file_path, 'w') as f:
    json.dump(result_dict, f)
