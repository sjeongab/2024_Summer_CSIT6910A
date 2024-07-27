import torch
import os
import numpy as np
from typing import Dict, Literal
from utils.system_utils import mkdir_p
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.models.base_model import Model, ModelConfig
from dataclasses import dataclass, field
from typing import Dict, List, Type, Literal, Tuple

@dataclass
class SeathruModelConfig(ModelConfig):
    """SeaThru-NeRF Config."""

    _target: Type = field(default_factory=lambda: MediumModel)
    near_plane: float = 0.05
    far_plane: float = 10.0
    num_levels: int = 16
    min_res: int = 16
    max_res: int = 8192
    log2_hashmap_size: int = 21
    features_per_level: int = 2
    num_layers: int = 2
    hidden_dim: int = 256
    bottleneck_dim: int = 63
    num_layers_colour: int = 3
    hidden_dim_colour: int = 256
    num_layers_medium: int = 2
    hidden_dim_medium: int = 128
    implementation: Literal["tcnn", "torch"] = "tcnn"
    use_viewing_dir_obj_rgb: bool = False
    object_density_bias: float = 0.0
    medium_density_bias: float = 0.0
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 128) #can be reduced?
    num_nerf_samples_per_ray: int = 64 #can be reduced?
    proposal_update_every: int = 5
    proposal_warmup: int = 5000
    num_proposal_iterations: int = 2
    use_same_proposal_network: bool = False
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 512,
                "use_linear": False,
            },
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 7,
                "max_res": 2048,
                "use_linear": False,
            },
        ]
    )
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    interlevel_loss_mult: float = 1.0
    use_proposal_weight_anneal: bool = True
    proposal_weights_anneal_slope: float = 10.0
    proposal_weights_anneal_max_num_iters: int = 15000
    use_single_jitter: bool = True
    disable_scene_contraction: bool = False
    use_gradient_scaling: bool = False
    initial_acc_loss_mult: float = 0.0001
    final_acc_loss_mult: float = 0.0001
    acc_decay: int = 10000
    rgb_loss_use_bayer_mask: bool = False
    prior_on: Literal["weights", "transmittance"] = "transmittance"
    debug: bool = False
    beta_prior: float = 100.0
    use_viewing_dir_obj_rgb: bool = False
    use_new_rendering_eqs: bool = True


class MediumField(Field):
    def __init__(self,
                 implementation: Literal["tcnn", "torch"] = "tcnn",
                 num_layers_medium: int = 2,
                 hidden_dim_medium: int = 128,
                 medium_density_bias: float = 0.0,
                 ):
        super().__init__()

        self.medium_colour = torch.tensor([1/255, 50/255, 32/255], device="cuda")
        self.medium_density_bias = medium_density_bias
        self.direction_encoding = SHEncoding(levels=4, implementation=implementation)
        self.colour_activation = torch.nn.Sigmoid()
        self.sigma_activation = torch.nn.Softplus()
        self.medium_mlp = MLP(
            in_dim = self.direction_encoding.get_out_dim(),
            num_layers=num_layers_medium,
            layer_width=hidden_dim_medium,
            out_dim=6,
            activation=torch.nn.Softplus(),
            out_activation=None,
            implementation=implementation,
        )

    def calculate_directions(self, camera):
        W = camera.image_width
        H = camera.image_height
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = torch.tensor(np.stack([(i-W*.5)/camera.FoVx, -(j-H*.5)/camera.FoVy, -np.ones_like(i)], -1), device="cuda")#torch.tensor(np.stack([(i-W*.5)/camera.FoVx, -(j-H*.5)/camera.FoVy, -np.ones_like(i)], -1), device="cuda")
        rays_d = torch.sum(dirs[..., np.newaxis, :] * camera.world_view_transform[:3, :3], -1)

        directions = get_normalized_directions(torch.nn.functional.normalize(rays_d))
        directions_flat = directions.view(-1, 3)
        directions_encoded = self.direction_encoding(directions_flat)
        return directions_encoded

    def get_outputs(self, ray_samples):
        #TODO: add cam_dir encoding?

        outputs = {}

        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        directions_encoded = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # Medium MLP forward pass
        medium_base_out = self.medium_mlp(directions_encoded)

        # different activations for different outputs
        medium_rgb = (
            self.colour_activation(medium_base_out[..., :3])
            .view(*outputs_shape, -1)
            .to(directions)
        )
        medium_bs = (
            self.sigma_activation(medium_base_out[..., 3:6] + self.medium_density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )
        """medium_attn = (
            self.sigma_activation(medium_base_out[..., 6:] + self.medium_density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )"""

        outputs["MEDIUM_RGB"] = medium_rgb
        outputs["MEDIUM_BS"] = medium_bs
        #outputs[SeathruHeadNames.MEDIUM_ATTN] = medium_attn

        return outputs

    
    def forward(self, camera):
        """base_field"""
        """if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding = self.get_density(ray_samples)"""
        
        directions = self.calculate_directions(camera)
        field_outputs = self.get_outputs(directions)

        return field_outputs    
    
    def save(self, path):
        mkdir_p(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class MediumModel(Model):
    """Seathru model

    Args:
        config: SeaThru-NeRF configuration to instantiate the model with.
    """

    config: SeathruModelConfig  # type: ignore

    def populate_modules(self):
        """Setup the fields and modules."""
        super().populate_modules()

        # Initialize SeaThru field
        self.field = MediumField(
            aabb=self.scene_box.aabb,
            num_levels=self.config.num_levels,
            min_res=self.config.min_res,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            features_per_level=self.config.features_per_level,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            bottleneck_dim=self.config.bottleneck_dim,
            #num_layers_colour=self.config.num_layers_colour,
            #hidden_dim_colour=self.config.hidden_dim_colour,
            num_layers_medium=self.config.num_layers_medium,
            hidden_dim_medium=self.config.hidden_dim_medium,
            spatial_distortion=None,
            implementation=self.config.implementation,
            #use_viewing_dir_obj_rgb=self.config.use_viewing_dir_obj_rgb,
            #object_density_bias=self.config.object_density_bias,
            medium_density_bias=self.config.medium_density_bias,
        )

        # Initialize proposal network(s) (this code snippet is taken from from nerfacto)
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert (
                len(self.config.proposal_net_args_list) == 1
            ), "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[
                    min(i, len(self.config.proposal_net_args_list) - 1)
                ]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.density_fn for network in self.proposal_networks]
            )

        def update_schedule(step):
            return np.clip(
                np.interp(
                    step,
                    [0, self.config.proposal_warmup],
                    [0, self.config.proposal_update_every],
                ),
                1,
                self.config.proposal_update_every,
            )

        # Initial sampler
        initial_sampler = None  # None is for piecewise as default
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(
                single_jitter=self.config.use_single_jitter
            )

        # Proposal sampler
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane, far_plane=self.config.far_plane
        )

        # Renderers
        self.renderer_rgb = SeathruRGBRenderer(
            use_new_rendering_eqs=self.config.use_new_rendering_eqs
        )
        self.renderer_depth = SeathruDepthRenderer(
            far_plane=self.config.far_plane, method="median"
        )
        self.renderer_accumulation = AccumulationRenderer()

        # Losses
        self.rgb_loss = MSELoss(reduction="none")

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Step member variable to keep track of the training step
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the parameter groups for the optimizer. (Code snippet from nerfacto)

        Returns:
            The parameter groups.
        """
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def step_cb(self, step) -> None:
        """Function for training callbacks to use to update training step.

        Args:
            step: The training step.
        """
        self.step = step

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Get the training callbacks.
           (Code of this function is from nerfacto but added step tracking for debugging.)

        Args:
            training_callback_attributes: The training callback attributes.

        Returns:
            List with training callbacks.
        """
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

        # Additional callback to track the training step for decaying and
        # debugging purposes
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.step_cb,
            )
        )

        return callbacks

    def get_outputs(self, ray_bundle):
        ray_samples: RaySamples

        # Get output from proposal network(s)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )

        # Get output from Seathru field
        field_outputs = self.field.forward(ray_samples)
        field_outputs[FieldHeadNames.DENSITY] = torch.nan_to_num(
            field_outputs[FieldHeadNames.DENSITY], nan=1e-3
        )
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        if self.training or not self.config.use_new_rendering_eqs:
            rgb = self.renderer_rgb(
                object_rgb=field_outputs[FieldHeadNames.RGB],
                medium_rgb=field_outputs[SeathruHeadNames.MEDIUM_RGB],  # type: ignore
                medium_bs=field_outputs[SeathruHeadNames.MEDIUM_BS],  # type: ignore
                medium_attn=field_outputs[SeathruHeadNames.MEDIUM_ATTN],  # type: ignore
                densities=field_outputs[FieldHeadNames.DENSITY],
                weights=weights,
                ray_samples=ray_samples,
            )
            direct = None
            bs = None
            J = None
        else:
            rgb, direct, bs, J = self.renderer_rgb(
                object_rgb=field_outputs[FieldHeadNames.RGB],
                medium_rgb=field_outputs[SeathruHeadNames.MEDIUM_RGB],  # type: ignore
                medium_bs=field_outputs[SeathruHeadNames.MEDIUM_BS],  # type: ignore
                medium_attn=field_outputs[SeathruHeadNames.MEDIUM_ATTN],  # type: ignore
                densities=field_outputs[FieldHeadNames.DENSITY],
                weights=weights,
                ray_samples=ray_samples,
            )

        # Render depth and accumulation
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # Calculate transmittance and add to outputs for acc_loss calculation
        # Ignore type error that occurs because ray_samples can be initialized without deltas
        transmittance = get_transmittance(
            ray_samples.deltas, field_outputs[FieldHeadNames.DENSITY]  # type: ignore
        )
        outputs = {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
            "transmittance": transmittance,
            "weights": weights,
            "direct": direct if not self.training else None,
            "bs": bs if not self.training else None,
            "J": J if not self.training else None,
        }

        # Add outputs from proposal network(s) to outputs if training for proposal loss
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        # Add proposed depth to outputs
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)

        if self.config.rgb_loss_use_bayer_mask:
            bayer_mask = get_bayer_mask(batch["indices"][:, 1:].to(self.device))
            squared_error = self.rgb_loss(image, outputs["rgb"])  # clip or not clip?
            scaling_grad = 1 / (outputs["rgb"].detach() + 1e-3)
            loss = squared_error * torch.square(scaling_grad)
            denom = torch.sum(bayer_mask)
            loss_dict["rgb_loss"] = torch.sum(loss * bayer_mask) / denom
        else:
            loss_dict["rgb_loss"] = recon_loss(gt=image, pred=outputs["rgb"])

        if self.training:
            if self.step < self.config.acc_decay:
                acc_loss_mult = self.config.initial_acc_loss_mult
            else:
                acc_loss_mult = self.config.final_acc_loss_mult

            if self.config.prior_on == "weights":
                loss_dict["acc_loss"] = acc_loss_mult * acc_loss(
                    transmittance_object=outputs["weights"], beta=self.config.beta_prior
                )
            elif self.config.prior_on == "transmittance":
                loss_dict["acc_loss"] = acc_loss_mult * acc_loss(
                    transmittance_object=outputs["transmittance"],
                    beta=self.config.beta_prior,
                )
            else:
                raise ValueError(f"Unknown prior_on: {self.config.prior_on}")

            # Proposal loss
            loss_dict[
                "interlevel_loss"
            ] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Get evaluation metrics dictionary and images to log for eval batch.
        (extended from nerfacto)

        Args:
            outputs: Dict containing the outputs of the model.
            batch: Dict containing the gt data.

        Returns:
            Tuple containing the metrics to log (as scalars) and the images to log.
        """
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]

        # Accumulation and depth maps
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Log the images
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        if self.config.use_new_rendering_eqs:
            # J (clean image), direct and bs images
            direct = outputs["direct"]
            bs = outputs["bs"]
            J = outputs["J"]

            combined_direct = torch.cat([direct], dim=1)
            combined_bs = torch.cat([bs], dim=1)
            combined_J = torch.cat([J], dim=1)

            # log the images
            images_dict["direct"] = combined_direct
            images_dict["bs"] = combined_bs
            images_dict["J"] = combined_J

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # Compute metrics
        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # Log the metrics (as scalars)
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        # Log the proposal depth maps
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(outputs[key])
            images_dict[key] = prop_depth_i

        # Debugging
        if self.config.debug:
            save_debug_info(
                weights=outputs["weights"],
                transmittance=outputs["transmittance"],
                depth=outputs["depth"],
                prop_depth=outputs["prop_depth_0"],
                step=self.step,
            )

        return metrics_dict, images_dict
    