import os.path as osp
import os
import cv2
import torch
import torch.utils.data as data
import numpy as np
import PIL.Image as Image
import random
import json

__all__ = [
    'GCNDataLayer',
]


class GCNDataLayer(data.Dataset):
    def __init__(self, data_root, cause, sessions, time_steps, camera_transforms, data_augmentation=False, dist=False, training=True):
        self.width = 1280
        self.height = 720
        self.num_box = 60
        self.data_root = data_root
        self.cause = cause
        self.sessions = sessions
        self.time_steps = time_steps
        self.training = training
        self.camera_transforms = camera_transforms
        self.data_augmentation = data_augmentation
        self.dist = dist

        self.inputs = []
        cnt_go, cnt_stop = 0, 0

        for session in self.sessions:
            # load positive and negative samples of this session
            positive_frame, negative_frame, positive_frame2 = self.behavior_gt(
                session)

            # go
            if len(positive_frame) > 2:
                st, et = positive_frame[0], positive_frame[-1]
                for start, end in zip(range(st, et, self.time_steps), range(st + self.time_steps, et, self.time_steps)):
                    self.inputs.append([session, start, end, np.array([0])])
                    cnt_go += 1

            # stop
            if len(negative_frame) > 2:
                st, et = negative_frame[0], negative_frame[-1]
                for start, end in zip(range(st, et, self.time_steps), range(st + self.time_steps, et, self.time_steps)):
                    self.inputs.append([session, start, end, np.array([1])])
                    cnt_stop += 1

            # go
            if len(positive_frame2) > 2:
                st, et = positive_frame2[0], positive_frame2[-1]
                for start, end in zip(range(st, et, self.time_steps), range(st + self.time_steps, et, self.time_steps)):
                    self.inputs.append([session, start, end, np.array([0])])
                    cnt_go += 1

        print(f"Label 'go':   {cnt_go:4d}")
        print(f"Label 'stop': {cnt_stop:4d}")
        # Training: {'go': 17707, 'stop': 3912}
        # Evaluating: {'go': 2238, 'stop':  316}

    def behavior_gt(self, session):

        rgb_path = osp.join(session, 'rgb/front')
        all_frame = sorted(os.listdir(rgb_path))
        # n_frame = len(all_frame)
        first_frame_id = int(all_frame[0].split('.')[0])
        last_frame_id = int(all_frame[-1].split('.')[0])

        annotations_path = session.split('variant_scenario')[
            0]+'behavior_annotation.txt'
        file = open(annotations_path, 'r')
        temp = file.readlines()
        file.close()

        st = int(temp[0].strip())
        ed = int(temp[1].strip())

        if last_frame_id <= ed:
            ed = last_frame_id

        return list(range(first_frame_id, st)), list(range(st, ed+1)), list(range(ed+1, last_frame_id+1))

    def normalize_box(self, trackers):
        trackers[:, :, 3] = trackers[:, :, 1] + trackers[:, :, 3]
        trackers[:, :, 2] = trackers[:, :, 0] + trackers[:, :, 2]

        tmp = trackers[:, :, 0] / self.width
        trackers[:, :, 0] = trackers[:, :, 1] / self.height
        trackers[:, :, 1] = tmp
        tmp = trackers[:, :, 2] / self.width
        trackers[:, :, 2] = trackers[:, :, 3] / self.height
        trackers[:, :, 3] = tmp

        return trackers

    def cal_pairwise_distance(self, X):
        """
        computes pairwise distance between each element
        Args:
            X: [N,D]
            Y: [M,D]
        Returns:
            dist: [N,M] matrix of euclidean distances
        """

        Y = X
        X = np.expand_dims(X[0], axis=0)
        rx = np.reshape(np.sum(np.power(X, 2), axis=1), (-1, 1))
        ry = np.reshape(np.sum(np.power(Y, 2), axis=1), (-1, 1))
        dist = np.clip(rx-2.0*np.matmul(X, np.transpose(Y)) +
                       np.transpose(ry), 0.0, float('inf'))

        return np.sqrt(dist)

    def compute_dist(self, tracker, depth, num_object):
        ##################################
        # tracker: Nx4 y1,x1,y2,x2
        #
        ##################################
        self._inv_intrinsics = np.linalg.inv(
            np.array([[936.86, 0.0, 647.48], [0.0, 936.4, 404.14], [0.0, 0.0, 1.0]]))
        threshold = 5

        center_x = np.expand_dims(
            (tracker[:, 1]+tracker[:, 3])*0.5*self.width, axis=1)  # Nx1
        center_y = np.expand_dims(
            (tracker[:, 0]+tracker[:, 2])*0.5*self.height, axis=1)  # Nx1

        center = np.concatenate([center_x, center_y], axis=1)  # Nx2
        depth_list = depth[center_y.astype(
            np.int32), center_x.astype(np.int32)]  # (N,)

        depth_list[0] = 1.0
        depth_list = np.reshape(depth_list, [-1])
        center[0, :] = np.array([640, 719])  # ego position
        center = np.append(center, np.ones(
            [np.shape(center)[0], 1]).astype(np.float32), axis=1)  # N*3
        center_3d = np.multiply(
            np.matmul(self._inv_intrinsics, np.transpose(center)), depth_list)
        distance_map = self.cal_pairwise_distance(np.transpose(center_3d))
        distance_mask = np.ones(self.num_box)
        zero_index = np.where(distance_map > threshold)
        distance_mask[zero_index[1]] = 0
        distance_mask[num_object+1:] = 0

        return distance_mask

    def process_tracking(self, session, start, end):

        tracking_results = np.load(osp.join(session, 'tracking.npy'))
        """
        tracking_results = np.array(
            [[187, 876, 1021, 402, 259, 317, 1, -1, -1, -1]])
        """

        try:
            t_array = tracking_results[:, 0]

        except Exception as e:
            print(session)
            print(tracking_results.shape)
            print(tracking_results)
            exit()

        tracking_index = tracking_results[np.where(t_array == end)[0], 1]
        num_object = len(tracking_index)

        trackers = np.zeros([self.time_steps, self.num_box, 4])   # TxNx4
        trackers[:, 0, :] = np.array(
            [0.0, 0.0, self.width, self.height])  # Ego bounding box

        for t in range(start, end):
            current_tracking = tracking_results[np.where(t_array == t+1)[0]]
            for i, object_id in enumerate(tracking_index):
                if i > self.num_box - 2:
                    break
                if object_id in current_tracking[:, 1]:
                    bbox = current_tracking[np.where(
                        current_tracking[:, 1] == object_id)[0], 2:6]

                    trackers[t-start, i+1, :] = bbox

        trackers = self.normalize_box(trackers)  # TxNx4 : y1, x1, y2, x2

        return trackers, num_object

    def __getitem__(self, index):

        session, start, end, vel_target = self.inputs[index]

        camera_inputs = []
        for idx in range(start, end):
            camera_name = str(idx).zfill(8)+'.png'
            # camera_path = osp.join(self.data_root, 'camera', session, camera_name)
            camera_path = osp.join(session, 'rgb/front', camera_name)
            img = self.camera_transforms(
                Image.open(camera_path).convert('RGB'))
            img = np.array(img)
            camera_inputs.append(img)

        camera_inputs = np.stack(camera_inputs)
        camera_inputs = torch.from_numpy(
            camera_inputs.astype(np.float32))  # (t, c, w, h)

        trackers, num_object = self.process_tracking(session, start, end)

        dist_mask = np.ones(self.num_box)
        if self.dist:
            # load depth result
            depth = 1.0 / (cv2.imread(osp.join(self.data_root, 'depth_midas',
                                               session, str(end).zfill(5) + '.png'), 0) / 255.0 + np.exp(-10))
            dist_mask = self.compute_dist(
                trackers[-1, :, ], depth, num_object)  # Nx4 : y1,x1,y2,x2

        dist_mask = torch.from_numpy(dist_mask.astype(np.float32))

        # add data augmentation
        if self.data_augmentation and vel_target[0] == 0:
            obj_id = random.randint(0, num_object*2)

            if obj_id < num_object:
                obj_id += 1
                # trackers[:, obj_id, :] = 0
                mask = np.ones((self.time_steps, 3, 299, 299))

                for t in range(self.time_steps):
                    x1, y1, x2, y2 = trackers[t, obj_id, :]
                    x1 = int(x1*299)  # x1
                    x2 = int(x2*299)  # x2
                    y1 = int(y1*299)  # y1
                    y2 = int(y2*299)  # y2
                    mask[t, :, x1:x2, y1:y2] = 0

                mask = torch.from_numpy(mask.astype(np.float32))

            else:
                mask = torch.ones((self.time_steps, 3, 299, 299))

        else:
            mask = torch.ones((self.time_steps, 3, 299, 299))

        trackers = torch.from_numpy(trackers.astype(np.float32))
        vel_target = torch.from_numpy(vel_target.astype(np.int64))

        return camera_inputs, trackers, mask, dist_mask, vel_target

    def __len__(self):
        return len(self.inputs)
