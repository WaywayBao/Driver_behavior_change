import os.path as osp
import os
from collections import OrderedDict
import socket
import getpass
machine_name = socket.gethostname()
username = getpass.getuser()

__all__ = ['parse_args']


def parse_args(parser):
    parser.add_argument(
        '--data_root', default=osp.expanduser('/home/cli/hdd'), type=str)
    parser.add_argument(
        '--save_path', default=osp.expanduser('/home/cli/exp_trn'), type=str)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--test_interval', default=1, type=int)
    parser.add_argument('--width', default=1280, type=int)
    parser.add_argument('--height', default=720, type=int)

    args = parser.parse_args()
    args.data = osp.basename(osp.normpath(args.data_root))

    if args.data == 'hdd':
        args.data = 'HDD'

    args.data_root = '/media/waywaybao_cs10/Disk_2/Retrieve_tool/data_collection/interactive'
    args.save_path = '/home/carla_data/save'

    args.class_index = list(data_info[args.data]['class_info'].keys())
    args.class_weight = list(data_info[args.data]['class_info'].values())

    args.test_session_set = []
    args.train_session_set = []

    for basic_scene in os.listdir(args.data_root):
        basic_scene_path = osp.join(
            args.data_root, basic_scene, 'variant_scenario')

        for var_scene in os.listdir(basic_scene_path):
            var_scene_path = osp.join(basic_scene_path, var_scene)


            if basic_scene[:2] == '10':
                args.test_session_set.append(var_scene_path)
            else:
                args.train_session_set.append(var_scene_path)

    args.num_classes = len(args.class_index)

    return args


data_info = OrderedDict()
data_info['HDD'] = OrderedDict()
'''
data_info['HDD']['class_info'] = OrderedDict([
    ('background',               1.0),
    ('intersection passing',     1.0),
    ('left turn',                1.0),
    ('right turn',               1.0),
    ('left lane change',         1.0),
    ('right lane change',        1.0),
    ('left lane branch',         1.0),
    ('right lane branch',        1.0),
    ('crosswalk passing',        1.0),
    ('railroad passing',         1.0),
    ('merge',                    1.0),
    ('U-turn',                   1.0),
])
'''
data_info['HDD']['class_info'] = OrderedDict([
    ('go',       1.0),
    ('stop',     1.0),
])

