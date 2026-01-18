import argparse


class Config:
    def __init__(self):
        self.mode = 'train'
        self.eval_full = False
        self.eval_path = None
        self.mark = None
        # self.eval_physics = False
        # self.eval_with_simulator = False

        #* training
        self.random_seed = None
        self.gradient_accumulation_steps = None
        self.gradient_clip = 1.0
        self.max_epochs = None
        self.output_dir = None
        self.optimizer = None
        self.base_learning_rate = None
        self.scheduler = None
        self.gamma = None
        self.lr_step = None
        self.checkpoint = None
        self.pretrain = None
        self.remove_pretrained_keys = []
        self.print_freq = None
        self.viz_freq = 500
        self.full_evaluation_freq = 5
        self.start_with_eval = False
        self.use_mix_trainset = False

        #* dataset
        self.dataset_name = None
        self.data_dir = None
        self.clean_data_mode = None
        self.img_size = (640, 480)
        self.bbox_scale_factor = None
        self.patch_size = None
        self.batch_size = None
        self.eval_batch_size = None
        self.num_workers = None
        self.eval_num_workers = None
        self.contact_normal_distance_thresh = (-0.01, 0.01)
        self.contact_vertical_distance_thresh = 0.005

        #* augmentation
        self.center_jittering = None
        self.scale_factor = None
        self.max_rot = None
        self.rot_prob = None
        self.clahe_prob = None
        self.RGB_shift_prob = None
        self.shift_limit = None
        self.color_jitter_prob = None
        self.brightness = None
        self.contrast = None
        self.saturation = None
        self.hue = None
        self.gaussian_blur_prob = None
        self.blur_limit = None
        self.sigma_limit = None
        self.motion_blur_prob = None
        self.motion_blur_limit = None
        self.random_erasing_prob = None
        self.random_erasing_mode = None
        self.random_erasing_min_area = None
        self.random_erasing_max_area = None
        self.random_erasing_max_count = None

        #* model
        self.model = None
        self.sde_mode = None
        self.repeat_num = 20
        self.sampler = 'ode'
        self.sampling_steps = 500
        self.eval_repeat_num = 50

        self.heatmap_size = 64
        self.heatmap_hand_sigma = 3.0
        self.heatmap_obj_sigma = 3.0
        self.roi_size = 32

        #* loss
        self.weight_diff_hand_loss = 1.0
        self.weight_diff_obj_loss = 1.0
        self.weight_hm_hand_loss = 1.0
        self.weight_hm_obj_loss = 1.0
        self.weight_segm_obj_loss = 1.0
        self.weight_vert_loss = 1.0
        self.weight_joint_loss = 1.0
        self.weight_mano_pose_loss = 1.0
        self.weight_mano_shape_loss = 1.0
        self.weight_hand_contact_loss = 1.0
        self.weight_force_loss = 1.0
        self.weight_gravity_loss = 1.0
        self.weight_torque_loss = 1.0
        self.weight_distrib_loss = 1.0
        self.weight_supervised_loss = 1.0
        self.weight_point_loss = 1.0
        self.weight_CoM_loss = 1.0
        self.weight_joint2hm_loss = 1.0
        self.weight_obj_reg_vert_loss = 1.0
        self.weight_obj_reg_kpt_loss = 1.0
        self.weight_obj_reg_rot6d_loss = 1.0
        self.weight_obj_reg_trans_loss = 1.0

        #* sample
        self.sample_T0 = 0.55
        self.sample_num = 50
        self.topk_hand = 10
        self.topk_obj = 5

        #* aggregation
        self.do_weighted_average = True
        self.do_physics_selection = True
        self.aggregation_mode_hand = 'heatmap_cascade'
        self.aggregation_mode_obj = 'heatmap_cascade'
        self.use_regression_as_candidate = True

def get_args():
    parser = argparse.ArgumentParser(description='Hand-Object Pose Estimation')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'infer'])
    parser.add_argument('--eval_full', action='store_true')
    parser.add_argument('--eval_path', type=str, default='')
    parser.add_argument('--mark', type=str, default='')

    #* training
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_clip', type=float, default=-1.)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam'])
    parser.add_argument('--base_learning_rate', type=float, default=2e-4)
    parser.add_argument('--scheduler', type=str, default='exp', choices=['exp', 'cosine', 'step'])
    parser.add_argument('--gamma', type=float, default=0.96)
    parser.add_argument('--lr_step', type=int, default=5)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--remove_pretrained_keys', nargs='+', default=[])
    parser.add_argument('--start_with_eval', action='store_true')
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--viz_freq', type=int, default=50)
    parser.add_argument('--full_evaluation_freq', type=int, default=10)
    
    #* dataset
    parser.add_argument('--dataset_name', type=str, default='dexycb', choices=['dexycb', 'ho3d'])
    parser.add_argument('--data_dir', type=str, default='/root/Workspace/HOI/data/DexYCB')
    parser.add_argument('--clean_data_mode', type=str, default='2023_CVPR_HFL', choices=['2023_CVPR_HFL', 
                                                                                         '2022_CVPR_ArtiBoost', 
                                                                                         '2023_WACV_DMA', 
                                                                                         'stable_grasping', 
                                                                                         '2023_NIPS_DeepSimHO'])
    parser.add_argument('--bbox_scale_factor', type=float, default=1.2)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--eval_num_workers', type=int, default=8)
    parser.add_argument('--use_mix_trainset', action='store_true')

    #* augmentation
    parser.add_argument('--center_jittering', type=float, default=0.2)
    parser.add_argument('--scale_factor', type=float, default=0.2)
    parser.add_argument('--max_rot', type=float, default=30)
    parser.add_argument('--rot_prob', type=float, default=1)
    parser.add_argument('--clahe_prob', type=float, default=0.5)
    parser.add_argument('--RGB_shift_prob', type=float, default=0.5)
    parser.add_argument('--shift_limit', type=float, default=(-20, 20))
    parser.add_argument('--color_jitter_prob', type=float, default=0.5)
    parser.add_argument('--brightness', type=float, default=(0.6, 1.3))
    parser.add_argument('--contrast', type=float, default=(0.6, 1.3))
    parser.add_argument('--saturation', type=float, default=(0.6, 1.3))
    parser.add_argument('--hue', type=float, default=(-0.15, 0.15))
    parser.add_argument('--gaussian_blur_prob', type=float, default=0.5)
    parser.add_argument('--blur_limit', type=float, default=(3, 7))
    parser.add_argument('--sigma_limit', type=float, default=(0.2, 2.0))
    parser.add_argument('--motion_blur_prob', type=float, default=0.5)
    parser.add_argument('--motion_blur_limit', type=float, default=(3, 7))
    parser.add_argument('--random_erasing_prob', type=float, default=0.5)
    parser.add_argument('--random_erasing_mode', type=str, default='pixel')
    parser.add_argument('--random_erasing_min_area', type=float, default=0.02)
    parser.add_argument('--random_erasing_max_area', type=float, default=0.2)
    parser.add_argument('--random_erasing_max_count', type=float, default=1)

    #* model
    parser.add_argument('--model',  type=str, default='vpho_net',
                        choices=['vpho_net',])
    parser.add_argument('--sde_mode', type=str, choices=['edm', 've', 'vp', 'subvp'], default='ve')
    parser.add_argument('--repeat_num', type=int, default=20)
    parser.add_argument('--sampler', type=str, choices=['ode'], default='ode')
    parser.add_argument('--sampling_steps', type=int, default=500)
    parser.add_argument('--eval_repeat_num', type=int, default=50)

    parser.add_argument('--heatmap_size', type=int, default=64) # checked
    parser.add_argument('--heatmap_hand_sigma', type=float, default=2.0) # checked
    parser.add_argument('--heatmap_obj_sigma', type=float, default=2.0) # checked
    parser.add_argument('--roi_size', type=int, default=32)

    #* loss
    parser.add_argument('--weight_diff_hand_loss', type=float, default=1.0)
    parser.add_argument('--weight_diff_obj_loss', type=float, default=1.0)
    parser.add_argument('--weight_hm_hand_loss', type=float, default=1e3)
    parser.add_argument('--weight_hm_obj_loss', type=float, default=1e3)
    parser.add_argument('--weight_segm_obj_loss', type=float, default=3e1)
    parser.add_argument('--weight_vert_loss', type=float, default=1e4)
    parser.add_argument('--weight_joint_loss', type=float, default=1e4)
    parser.add_argument('--weight_mano_pose_loss', type=float, default=10)
    parser.add_argument('--weight_mano_shape_loss', type=float, default=1.0)
    parser.add_argument('--weight_hand_contact_loss', type=float, default=1.0)
    parser.add_argument('--weight_force_loss', type=float, default=1.0)
    parser.add_argument('--weight_gravity_loss', type=float, default=1.0)
    parser.add_argument('--weight_torque_loss', type=float, default=30.0)
    parser.add_argument('--weight_supervised_loss', type=float, default=10)
    parser.add_argument('--weight_point_loss', type=float, default=1e2)
    parser.add_argument('--weight_CoM_loss', type=float, default=1e2)
    parser.add_argument('--weight_joint2hm_loss', type=float, default=1e3)
    parser.add_argument('--weight_obj_reg_vert_loss', type=float, default=1e4)
    parser.add_argument('--weight_obj_reg_kpt_loss', type=float, default=1e4)
    parser.add_argument('--weight_obj_reg_rot6d_loss', type=float, default=10)
    parser.add_argument('--weight_obj_reg_trans_loss', type=float, default=1e4)


    #* sample
    parser.add_argument('--sample_T0', type=float, default=0.65)
    parser.add_argument('--sample_num', type=int, default=50)
    parser.add_argument('--topk_hand', type=int, default=15)
    parser.add_argument('--topk_obj', type=int, default=5)

    #* aggregation
    parser.add_argument('--do_weighted_average', action='store_false')
    parser.add_argument('--do_physics_selection', action='store_false')
    parser.add_argument('--aggregation_mode_hand', type=str, default='heatmap_cascade', choices=['heatmap_cascade', 
                                                                                                 'heatmap', 
                                                                                                 '2D_pt_pose', 
                                                                                                 '2D_pt_joint',
                                                                                                 'average_all',
                                                                                                 'random'])
    parser.add_argument('--aggregation_mode_obj', type=str, default='heatmap_cascade', choices=['heatmap_cascade', 
                                                                                                'heatmap', 
                                                                                                '2D_pt_pose',
                                                                                                'average_all',
                                                                                                'random'])
    parser.add_argument('--use_regression_as_candidate', action='store_false')

    args = parser.parse_args()
    return args


args = get_args()
cfg = Config()


for k, v in vars(args).items():
    if hasattr(cfg, k):
        setattr(cfg, k, v)
    else:
        raise ValueError(f"Invalid config key: {k}")

