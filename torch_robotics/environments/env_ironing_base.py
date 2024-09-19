from copy import copy

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField, MultiRoundedBoxField, MultiSphereField
from torch_robotics.robots import RobotDenso
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


def create_table_object_field(tensor_args=None):
    centers = [(0., 0., 0.)]
    sizes = [(0.56, 0.90, 0.80)]
    centers = np.array(centers)
    sizes = np.array(sizes)
    boxes = MultiBoxField(centers, sizes, tensor_args=tensor_args)
    return ObjectField([boxes], 'table')


def create_board_field(tensor_args=None):
    width1 = 0.19
    width2 = 0.38
    depth = 0.04
    height1 = 0.15
    height2 = 0.56
    center1 = [(0.5, 0.73, 0.90 + height1/2)]
    center2 = [(0.5, 0.73, 0.90 - height2/2)]


    # top board
    centers = center1
    sizes = [(width1, depth, height1)]
    # bottom board
    centers+= center2
    sizes.append((width2, depth, height2))

    # frame
    frame_width = 0.02
    frame_depth = 0.02
    frame_height = 0.90
    frame_centers = [(0.5, 0.73 + frame_depth/2 + depth/2, 0.45)]
    centers += frame_centers
    sizes.append((frame_width, frame_depth, frame_height))


    centers = np.array(centers)
    sizes = np.array(sizes)

    boxes = MultiBoxField(centers, sizes, tensor_args=tensor_args)
    return ObjectField([boxes], 'shelf')


class EnvIroningBase(EnvBase):

    def __init__(self, tensor_args=None, **kwargs):
        # table object field
        # table_obj_field = create_table_object_field(tensor_args=tensor_args)
        # table_sizes = table_obj_field.fields[0].sizes[0]
        # dist_robot_to_table = 0.10
        # theta = np.deg2rad(90)
        # table_obj_field.set_position_orientation(
        #     pos=(dist_robot_to_table + table_sizes[1].item()/2, 0, -table_sizes[2].item()/2),
        #     ori=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
        # )

        # shelf object field
        board_obj_field = create_board_field(tensor_args=tensor_args)
        # theta = np.deg2rad(-90)
        board_obj_field.set_position_orientation(
            pos=(0,0,0),
            # ori=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
        )

        obj_list = [board_obj_field]

        super().__init__(
            name=self.__class__.__name__,
            limits=torch.tensor([[-1, -1, -1], [1.5, 1., 1.5]], **tensor_args),  # environments limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=250,
            num_samples=64,
            sigma_start=1e-3,
            sigma_gp=1e-1,
            sigma_goal_prior=1e-3,
            sigma_coll=1e-4,
            step_size=5e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.1,
            sigma_start_sample=1e-3,
            sigma_goal_sample=1e-3,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )
        if isinstance(robot, RobotDenso):
            return params
        else:
            raise NotImplementedError

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=torch.pi/80,
            n_radius=torch.pi/4,
            n_pre_samples=50000,
            max_time=15
        )
        if isinstance(robot, RobotDenso):
            return params
        else:
            raise NotImplementedError


if __name__ == '__main__':
    env = EnvIroningBase(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    # env.render_grad_sdf(ax, fig)
    plt.show()
