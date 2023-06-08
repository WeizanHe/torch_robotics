import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environment.env_base import EnvBase
from torch_robotics.environment.primitives import ObjectField, MultiSphereField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvSpheres3D(EnvBase):

    def __init__(self, tensor_args=None, **kwargs):
        spheres = MultiSphereField(torch.tensor([
                    [0.6, 0.3, 0.],
                    [0.5, 0.3, 0.5],
                    [-0.5, 0.25, 0.6],
                    [-0.6, -0.2, 0.4],
                    [-0.7, 0.1, 0.0],
                    [0.5, -0.45, 0.2],
                    [0.6, -0.35, 0.6],
                    [0.3, 0.0, 1.0],
                    [0.0, -0.25, 0.25],
                    [0.0, 0.4, 0.3],
                    ]),
                torch.tensor([
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.15
                ]),
                tensor_args=tensor_args)

        obj_field = ObjectField([spheres], 'spheres')
        obj_list = [obj_field]

        super().__init__(
            name=self.__class__.__name__,
            limits=torch.tensor([[-1, -1, -1], [1, 1, 1]], **tensor_args),  # environment limits
            obj_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_gpmp_params(self):
        params = dict(
            opt_iters=100,
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
            sigma_gp_sample=5e-2,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )
        return params

    def get_rrt_connect_params(self):
        params = dict(
            n_iters=10000,
            step_size=torch.pi/80,
            n_radius=torch.pi/4,
            n_pre_samples=50000,
            max_time=15
        )
        return params


if __name__ == '__main__':
    env = EnvSpheres3D(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    env.render(ax)
    plt.show()