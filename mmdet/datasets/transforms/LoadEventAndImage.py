import random
from mmcv.transforms import BaseTransform
import torch
from mmdet.registry import TRANSFORMS
import numpy as np
import cv2

@TRANSFORMS.register_module()
class LoadEventAndImage(BaseTransform):
    """

    """

    def __init__(self, flow, event_source=0, transform_name='left_transformed', 
            fusion_module=dict(
                GSTFM=True,
                CC=False,
                EF=False,
                LF=False,
            ),
            dataset_name='DSEC',
            input_control='all',
            blank_ratio=-1,
        ):

        transform_name_dict = dict(

            # 相同视角
            left_transformed='transformed_images',
            LT='transformed_images',
            LTE='transformed_images',

            # 不等分辨率
            left_normal='left/rectified',
            LN='left/rectified',

            LNE='left_e_size',

            normal='right/rectified',
            right_normal='right/rectified',
            RN='right/rectified',
            
            rotation_45='rotation_45',
            R45='rotation_45',

            rotation_90='rotation_90',
            R90='rotation_90',

            rotation_180='rotation_180',
            R180='rotation_180',

            rotation_45_shear='rotation_45_shear',
            R45S='rotation_45_shear',

            rotation_90_shear='rotation_90_shear',
            R90S='rotation_90_shear',

            rotation_180_shear='rotation_180_shear',
            R180S='rotation_180_shear',
            

            # 相等分辨率
            normal_e_size='normal_e_size',
            right_normal_e_size='normal_e_size',
            RNE='normal_e_size',

            rotation_45_e_size='rotation_45_e_size',
            R45E='rotation_45_e_size',

            rotation_90_e_size='rotation_90_e_size',
            R90E='rotation_90_e_size',

            rotation_180_e_size='rotation_180_e_size',
            R180E='rotation_180_e_size',

            rotation_45_shear_e_size='rotation_45_shear_e_size',
            R45SE='rotation_45_shear_e_size',

            rotation_90_shear_e_size='rotation_90_shear_e_size',
            R90SE='rotation_90_shear_e_size',

            rotation_180_shear_e_size='rotation_180_shear_e_size',
            R180SE='rotation_180_shear_e_size',
        )

        self.only_event = flow['only_event']
        self.only_color = flow['only_color']
        self.fusion = flow['fusion']
        self.transform_name = transform_name_dict[transform_name]
        assert event_source in [0, 1]
        self.event_source = event_source

        assert self.only_event + self.only_color + self.fusion == 1, 'only one flow type can affect'

        self.fusion_module = fusion_module

        self.dataset_name = dataset_name

        self.input_control = input_control
        self.blank_ratio = blank_ratio

    def transform(self, results):
        """
        'data/DSEC/train/events/interlaken_00_c/events/event_images/000079.npy'
        
        """
        if self.dataset_name == 'DSEC':
            original_e_path = results['img_path']
            c_path = original_e_path.replace('events/', 'images/').replace('event_images/', self.transform_name + '/').replace('.npy', '.png')

            if self.event_source == 0:
                e_path = original_e_path
            elif self.event_source == 1:
                e_path = original_e_path.replace('event_images/', 'event_images_copy/')
        elif self.dataset_name == 'DSEC-Soft':
            original_c_path = results['img_path']
            if 'left/rectified' in original_c_path:
            # if LN GT
                c_path = original_c_path
                e_path = original_c_path.replace('images/', 'events/').replace('left/rectified/', 'event_images/').replace('.png', '.npy')
            elif 'right/rectified' in original_c_path:
            # if RN GT
                c_path = original_c_path
                e_path = original_c_path.replace('images/', 'events/').replace('right/rectified/', 'event_images/').replace('.png', '.npy')
            else:
            # if other view's GT
                c_path = original_c_path
                assert self.transform_name in c_path
                e_path = original_c_path.replace('/images/', '/events/').replace(self.transform_name, 'event_images').replace('.png', '.npy')



        if self.only_event:
            # event_image shape is (num_bins, H, W)
            e = np.load(e_path)
            # for data augmentation methods, need to permute event to (H, W, num_bins)
            e = np.transpose(e, (1, 2, 0))
            
            results['img'] = e
            
            results['ori_shape'] = e.shape[:2]
            results['img_shape'] = e.shape[:2]
            results['scale_factor'] = (1., 1.)
            
        elif self.only_color:
            c = cv2.imread(c_path)

            results['img'] = c
            results['img_path'] = c_path
            
            # normally, a image ndarray should be (H, W, 3)
            # in PackDetInputs, the image will do permute(2, 0, 1) to (3, H, W)
            # for detail, see PackDetInputs
            
            results['ori_shape'] = c.shape[:2]
            results['img_shape'] = c.shape[:2]
            results['scale_factor'] = (1., 1.)
        
        elif self.fusion:
            e = np.load(e_path)
            e = np.transpose(e, (1, 2, 0))

            c = cv2.imread(c_path)

        
            # input control
            if self.input_control == 'color':
                e = np.zeros_like(e)
            elif self.input_control == 'event':
                c = np.zeros_like(c)

            if self.blank_ratio > 0:
                if random.random() < self.blank_ratio:
                    c = np.zeros_like(c)
            


            if self.fusion_module['EF']:
                e = np.concatenate((e, c), axis=2)


            results['img'] = e
            results['c_img'] = c
            results['img_path'] = e_path
            results['c_img_path'] = c_path
            
            results['ori_shape'] = c.shape[:2]
            results['img_shape'] = c.shape[:2]
            results['scale_factor'] = (1., 1.)

        return results