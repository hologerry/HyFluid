import configargparse


def config_parser_density():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--basedir", type=str, default="./logs/", help="where to store ckpts and logs")
    parser.add_argument("--datadir", type=str, default="./data/llff/fern", help="input data directory")

    # training options
    parser.add_argument("--encoder", type=str, default="ingp", choices=["ingp", "plane"])
    parser.add_argument(
        "--N_rand", type=int, default=32 * 32 * 4, help="batch size (number of random rays per gradient step)"
    )
    parser.add_argument("--N_time", type=int, default=1, help="batch size in time")
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--lrate_decay", type=int, default=250, help="exponential learning rate decay")
    parser.add_argument("--N_iters", type=int, default=50000)
    parser.add_argument("--no_reload", action="store_true", help="do not reload weights from saved ckpt")
    parser.add_argument(
        "--ft_path", type=str, default=None, help="specific weights npy file to reload for coarse network"
    )

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, help="number of coarse samples per ray")
    parser.add_argument("--perturb", type=float, default=1.0, help="set to 0. for no jitter, 1. for jitter")

    parser.add_argument(
        "--render_only", action="store_true", help="do not optimize, reload weights and render out render_poses path"
    )
    parser.add_argument("--half_res", action="store_true", help="load at half resolution")

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help="frequency of console printout and metric logging")
    parser.add_argument("--i_weights", type=int, default=10000, help="frequency of weight ckpt saving")
    parser.add_argument("--i_video", type=int, default=9999999, help="frequency of render_poses video saving")

    parser.add_argument("--finest_resolution", type=int, default=512, help="finest resolution for hashed embedding")
    parser.add_argument("--finest_resolution_t", type=int, default=512, help="finest resolution for hashed embedding")
    parser.add_argument("--num_levels", type=int, default=16, help="number of levels for hashed embedding")
    parser.add_argument("--base_resolution", type=int, default=16, help="base resolution for hashed embedding")
    parser.add_argument("--base_resolution_t", type=int, default=16, help="base resolution for hashed embedding")
    parser.add_argument("--log2_hashmap_size", type=int, default=19, help="log2 of hashmap size")
    parser.add_argument("--feats_dim", type=int, default=36, help="feature dimension of kplanes")
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6, help="learning rate")

    parser.add_argument("--lrate_den", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--vel_path", type=str, default=None, help="specific weights npy file to reload for velocity network"
    )

    parser.add_argument("--train_vel", action="store_true", help="train velocity network")
    parser.add_argument("--run_advect_den", action="store_true", help="Run advect")
    parser.add_argument("--run_future_pred", action="store_true", help="Run future prediction")
    parser.add_argument("--run_view_synthesis", action="store_true", help="Run novel view synthesis (test views)")
    parser.add_argument(
        "--run_without_vort",
        action="store_true",
        help="by default we use vortex now, this flag will run without vortex",
    )
    parser.add_argument("--n_particles", type=int, default=100, help="how many particles to use")
    parser.add_argument("--sim_res_x", type=int, default=128, help="simulation resolution along X/width axis")
    parser.add_argument("--sim_res_y", type=int, default=192, help="simulation resolution along Y/height axis")
    parser.add_argument("--sim_res_z", type=int, default=128, help="simulation resolution along Z/depth axis")
    parser.add_argument(
        "--proj_y", type=int, default=128, help="projection resolution along Y/height axis, this must be 2**n"
    )
    parser.add_argument(
        "--y_start", type=int, default=48, help="Within sim_res_y, where to start the projection domain"
    )
    parser.add_argument("--use_project", action="store_true", help="use projection in re-simulation?")

    parser.add_argument("--finest_resolution_v", type=int, default=512, help="finest resolution for hashed embedding")
    parser.add_argument(
        "--finest_resolution_v_t", type=int, default=256, help="finest resolution for hashed embedding"
    )
    parser.add_argument("--base_resolution_v", type=int, default=16, help="base resolution for hashed embedding")
    parser.add_argument("--base_resolution_v_t", type=int, default=16, help="base resolution for hashed embedding")
    parser.add_argument("--no_vel_der", action="store_true", help="do not use velocity derivatives-related losses")
    parser.add_argument(
        "--save_fields", action="store_true", help="when run_advect_density, save fields for paraview rendering"
    )
    parser.add_argument("--vel_num_layers", type=int, default=2, help="number of layers in velocity network")
    parser.add_argument("--vel_scale", type=float, default=0.01)
    parser.add_argument("--vel_weight", type=float, default=0.1)
    parser.add_argument("--d_weight", type=float, default=0.1)
    parser.add_argument("--flow_weight", type=float, default=0.001)
    parser.add_argument("--vort_weight", type=float, default=1)
    parser.add_argument("--vort_intensity", type=float, default=20)
    parser.add_argument("--vort_radius", type=float, default=0.01)
    parser.add_argument("--rec_weight", type=float, default=0)
    parser.add_argument("--sim_steps", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--save_den", action="store_true", default=False)


    return parser


def config_parser_joint():

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--basedir", type=str, default="./logs/", help="where to store ckpts and logs")
    parser.add_argument("--datadir", type=str, default="./data/llff/fern", help="input data directory")

    # training options
    parser.add_argument(
        "--N_rand", type=int, default=32 * 32 * 4, help="batch size (number of random rays per gradient step)"
    )
    parser.add_argument("--N_time", type=int, default=1, help="batch size in time")
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--lrate_den", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--lrate_decay", type=int, default=250, help="exponential learning rate decay")
    parser.add_argument("--N_iters", type=int, default=5000)
    parser.add_argument("--no_reload", action="store_true", help="do not reload weights from saved ckpt")
    parser.add_argument(
        "--ft_path", type=str, default=None, help="specific weights npy file to reload for coarse network"
    )
    parser.add_argument(
        "--ft_v_path", type=str, default=None, help="specific weights npy file to reload for coarse network"
    )
    parser.add_argument("--use_f", action="store_true", default=False, help="predict f")
    parser.add_argument(
        "--detach_vel",
        action="store_true",
        default=False,
    )

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, help="number of coarse samples per ray")
    parser.add_argument("--perturb", type=float, default=1.0, help="set to 0. for no jitter, 1. for jitter")

    parser.add_argument(
        "--render_only", action="store_true", help="do not optimize, reload weights and render out render_poses path"
    )
    parser.add_argument("--train_vel", action="store_true", help="train velocity network")
    parser.add_argument("--run_advect_den", action="store_true", help="Run advect")
    parser.add_argument("--run_future_pred", action="store_true", help="Run future")
    parser.add_argument("--generate_vort_particles", action="store_true", help="shortcut to generate vort particles")
    parser.add_argument("--half_res", action="store_true", help="load at half resolution")
    parser.add_argument("--sim_res_x", type=int, default=128, help="simulation resolution along X/width axis")
    parser.add_argument("--sim_res_y", type=int, default=192, help="simulation resolution along Y/height axis")
    parser.add_argument("--sim_res_z", type=int, default=128, help="simulation resolution along Z/depth axis")
    parser.add_argument(
        "--proj_y", type=int, default=128, help="projection resolution along Y/height axis, this must be 2**n"
    )
    parser.add_argument(
        "--y_start", type=int, default=48, help="Within sim_res_y, where to start the projection domain"
    )
    parser.add_argument("--use_project", action="store_true", help="use projection in re-simulation?")

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help="frequency of console printout and metric logging")
    parser.add_argument("--i_weights", type=int, default=10000, help="frequency of weight ckpt saving")
    parser.add_argument("--i_video", type=int, default=9999999, help="frequency of render_poses video saving")

    parser.add_argument("--finest_resolution", type=int, default=512, help="finest resolution for hashed embedding")
    parser.add_argument("--finest_resolution_t", type=int, default=256, help="finest resolution for hashed embedding")
    parser.add_argument("--num_levels", type=int, default=16, help="number of levels for hashed embedding")
    parser.add_argument("--base_resolution", type=int, default=16, help="base resolution for hashed embedding")
    parser.add_argument("--base_resolution_t", type=int, default=16, help="base resolution for hashed embedding")
    parser.add_argument("--finest_resolution_v", type=int, default=512, help="finest resolution for hashed embedding")
    parser.add_argument(
        "--finest_resolution_v_t", type=int, default=256, help="finest resolution for hashed embedding"
    )
    parser.add_argument("--base_resolution_v", type=int, default=16, help="base resolution for hashed embedding")
    parser.add_argument("--base_resolution_v_t", type=int, default=16, help="base resolution for hashed embedding")
    parser.add_argument("--log2_hashmap_size", type=int, default=19, help="log2 of hashmap size")
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6, help="learning rate")
    parser.add_argument("--no_vel_der", action="store_true", help="do not use velocity derivatives-related losses")
    parser.add_argument(
        "--save_fields", action="store_true", help="when run_advect_density, save fields for preview rendering"
    )
    parser.add_argument("--save_den", action="store_true", help="for houdini rendering")
    parser.add_argument("--vel_num_layers", type=int, default=2, help="number of layers in velocity network")
    parser.add_argument("--vel_scale", type=float, default=0.01)
    parser.add_argument("--vel_weight", type=float, default=0.1)
    parser.add_argument("--d_weight", type=float, default=0.1)
    parser.add_argument("--flow_weight", type=float, default=0.001)
    parser.add_argument("--rec_weight", type=float, default=0)
    parser.add_argument("--sim_steps", type=int, default=1)
    parser.add_argument("--proj_weight", type=float, default=0.0)
    parser.add_argument("--d2v_weight", type=float, default=0.0)
    parser.add_argument("--coef_den2vel", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true", default=False)

    return parser




def config_parser_vort():

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--basedir", type=str, default="./logs/", help="where to store ckpts and logs")
    parser.add_argument("--datadir", type=str, default="./data/llff/fern", help="input data directory")

    # training options
    parser.add_argument(
        "--N_rand", type=int, default=32 * 32 * 4, help="batch size (number of random rays per gradient step)"
    )
    parser.add_argument("--N_time", type=int, default=1, help="batch size in time")
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--lrate_den", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--lrate_decay", type=int, default=250, help="exponential learning rate decay")
    parser.add_argument("--N_iters", type=int, default=5000)
    parser.add_argument("--no_reload", action="store_true", help="do not reload weights from saved ckpt")
    parser.add_argument(
        "--ft_path", type=str, default=None, help="specific weights npy file to reload for density network"
    )
    parser.add_argument(
        "--vel_path", type=str, default=None, help="specific weights npy file to reload for velocity network"
    )

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, help="number of coarse samples per ray")
    parser.add_argument("--perturb", type=float, default=1.0, help="set to 0. for no jitter, 1. for jitter")

    parser.add_argument("--train_vel", action="store_true", help="train velocity network")
    parser.add_argument("--run_advect_den", action="store_true", help="Run advect")
    parser.add_argument("--run_future_pred", action="store_true", help="Run future prediction")
    parser.add_argument("--run_view_synthesis", action="store_true", help="Run novel view synthesis (test views)")
    parser.add_argument(
        "--run_without_vort",
        action="store_true",
        help="by default we use vortex now, this flag will run without vortex",
    )
    parser.add_argument("--n_particles", type=int, default=100, help="how many particles to use")
    parser.add_argument("--half_res", action="store_true", help="load at half resolution")
    parser.add_argument("--sim_res_x", type=int, default=128, help="simulation resolution along X/width axis")
    parser.add_argument("--sim_res_y", type=int, default=192, help="simulation resolution along Y/height axis")
    parser.add_argument("--sim_res_z", type=int, default=128, help="simulation resolution along Z/depth axis")
    parser.add_argument(
        "--proj_y", type=int, default=128, help="projection resolution along Y/height axis, this must be 2**n"
    )
    parser.add_argument(
        "--y_start", type=int, default=48, help="Within sim_res_y, where to start the projection domain"
    )
    parser.add_argument("--use_project", action="store_true", help="use projection in re-simulation?")

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help="frequency of console printout and metric loggin")
    parser.add_argument("--i_weights", type=int, default=10000, help="frequency of weight ckpt saving")
    parser.add_argument("--i_video", type=int, default=9999999, help="frequency of render_poses video saving")

    parser.add_argument("--finest_resolution", type=int, default=512, help="finest resolution for hashed embedding")
    parser.add_argument("--finest_resolution_t", type=int, default=256, help="finest resolution for hashed embedding")
    parser.add_argument("--num_levels", type=int, default=16, help="number of levels for hashed embedding")
    parser.add_argument("--base_resolution", type=int, default=16, help="base resolution for hashed embedding")
    parser.add_argument("--base_resolution_t", type=int, default=16, help="base resolution for hashed embedding")
    parser.add_argument("--finest_resolution_v", type=int, default=512, help="finest resolution for hashed embedding")
    parser.add_argument(
        "--finest_resolution_v_t", type=int, default=256, help="finest resolution for hashed embedding"
    )
    parser.add_argument("--base_resolution_v", type=int, default=16, help="base resolution for hashed embedding")
    parser.add_argument("--base_resolution_v_t", type=int, default=16, help="base resolution for hashed embedding")
    parser.add_argument("--log2_hashmap_size", type=int, default=19, help="log2 of hashmap size")
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6, help="learning rate")
    parser.add_argument("--no_vel_der", action="store_true", help="do not use velocity derivatives-related losses")
    parser.add_argument(
        "--save_fields", action="store_true", help="when run_advect_density, save fields for paraview rendering"
    )
    parser.add_argument("--vel_num_layers", type=int, default=2, help="number of layers in velocity network")
    parser.add_argument("--vel_scale", type=float, default=0.01)
    parser.add_argument("--vel_weight", type=float, default=0.1)
    parser.add_argument("--d_weight", type=float, default=0.1)
    parser.add_argument("--flow_weight", type=float, default=0.001)
    parser.add_argument("--vort_weight", type=float, default=1)
    parser.add_argument("--vort_intensity", type=float, default=20)
    parser.add_argument("--vort_radius", type=float, default=0.01)
    parser.add_argument("--rec_weight", type=float, default=0)
    parser.add_argument("--sim_steps", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--save_den", action="store_true", default=False)

    return parser
