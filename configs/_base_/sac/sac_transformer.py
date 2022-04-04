_base_ = ['./sac.py']
stack_frame = 1
num_heads = 4

env_cfg = dict(
    type='gym',
    unwrapped=False,
    obs_mode='pointcloud',
    reward_type='dense',
    stack_frame=stack_frame
)

agent = dict(
    type='SAC',
    batch_size=1024,
    gamma=0.95,
    policy_cfg=dict(
        type='ContinuousPolicy',
        policy_head_cfg=dict(
            type='GaussianHead',
            log_sig_min=-20,
            log_sig_max=2,
            epsilon=1e-6
        ),
        nn_cfg=dict(
            type='PointNetWithInstanceInfoV0',
            stack_frame=stack_frame,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointNetV0',
                conv_cfg=dict(
                    type='ConvMLP',
                    norm_cfg=None,
                    mlp_spec=['agent_shape + pcd_xyz_rgb_channel', 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    conv_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[192 * stack_frame, 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                subtract_mean_coords=True,
                max_mean_mix_aggregation=True
            ),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape', 192, 192],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),                            
            transformer_cfg=dict(
                type='TransformerEncoder',
                block_cfg=dict(
                    attention_cfg=dict(
                        type='MultiHeadSelfAttention',
                        embed_dim=192,
                        num_heads=num_heads,
                        latent_dim=32,
                        dropout=0.1,
                    ),
                    mlp_cfg=dict(
                        type='LinearMLP',
                        norm_cfg=None,
                        mlp_spec=[192, 768, 192],
                        bias='auto',
                        inactivated_output=True,
                        linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                    ),
                    dropout=0.1,
                ),
                pooling_cfg=dict(
                    embed_dim=192,
                    num_heads=num_heads,
                    latent_dim=32,
                ),
                mlp_cfg=None,
                num_blocks=2,
            ),
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[192, 128, 'action_shape * 2'],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
        ),
        optim_cfg=dict(type='Adam', lr=3e-4, weight_decay=5e-6),
    ),
    value_cfg=dict(
        type='ContinuousValue',
        num_heads=2,
        nn_cfg=dict(
            type='PointNetWithInstanceInfoV0',
            stack_frame=stack_frame,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointNetV0',
                conv_cfg=dict(
                    type='ConvMLP',
                    norm_cfg=None,
                    mlp_spec=['agent_shape + pcd_xyz_rgb_channel + action_shape', 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    conv_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[192 * stack_frame, 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                subtract_mean_coords=True,
                max_mean_mix_aggregation=True
            ),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + action_shape', 192, 192],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),                            
            transformer_cfg=dict(
                type='TransformerEncoder',
                block_cfg=dict(
                    attention_cfg=dict(
                        type='MultiHeadSelfAttention',
                        embed_dim=192,
                        num_heads=num_heads,
                        latent_dim=32,
                        dropout=0.1,
                    ),
                    mlp_cfg=dict(
                        type='LinearMLP',
                        norm_cfg=None,
                        mlp_spec=[192, 768, 192],
                        bias='auto',
                        inactivated_output=True,
                        linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                    ),
                    dropout=0.1,
                ),
                pooling_cfg=dict(
                    embed_dim=192,
                    num_heads=num_heads,
                    latent_dim=32,
                ),
                mlp_cfg=None,
                num_blocks=2,
            ),
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[192, 128, 1],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
        ),
        optim_cfg=dict(type='Adam', lr=5e-4, weight_decay=5e-6),
    ),
)

expert = dict(
    type='SAC',
    batch_size=1024,
    gamma=0.95,
    policy_cfg=dict(
        type='ContinuousPolicy',
        policy_head_cfg=dict(
            type='GaussianHead',
            log_sig_min=-20,
            log_sig_max=2,
            epsilon=1e-6
        ),
        nn_cfg=dict(
            type='PointNetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + pcd_all_channel', 256, 512],
                bias='auto',
                inactivated_output=False,
                conv_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[512 * stack_frame, 256, 'action_shape * 2'],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True,
            stack_frame=stack_frame,
        ),
        optim_cfg=dict(type='Adam', lr=5e-4),
    ),
    value_cfg=dict(
        type='ContinuousValue',
        num_heads=2,
        nn_cfg=dict(
            type='PointNetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + action_shape + pcd_all_channel', 256, 512],
                bias='auto',
                inactivated_output=False,
                conv_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[512 * stack_frame, 256, 1],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True,
            stack_frame=stack_frame,
        ),
        optim_cfg=dict(type='Adam', lr=5e-4),
    ),
)

replay_cfg = dict(
    type='ReplayMemory',
    capacity=800000,
)

train_mfrl_cfg = dict(
    total_steps=2500000,
    warm_steps=4000,
    n_eval=2500000,
    n_checkpoint=100000,
    n_steps=8,
    n_updates=4,
    m_steps=8,
    init_replay_buffers=['./full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1000_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1000_link_1-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1001_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1002_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1006_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1007_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1007_link_1-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1014_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1017_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1018_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1018_link_1-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1025_link_1-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1026_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1027_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1028_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1030_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1031_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1034_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1034_link_1-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1036_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1036_link_1-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1038_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1039_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1041_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1042_link_1-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1042_link_3-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1044_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1044_link_1-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1044_link_2-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1044_link_3-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1045_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1046_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1047_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1049_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1051_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1052_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1052_link_1-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1054_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1057_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1060_link_2-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1060_link_3-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1061_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1062_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1063_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1064_link_1-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1064_link_3-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1065_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1067_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1068_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1073_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1075_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1075_link_1-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1077_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1078_link_0-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1078_link_1-v0.h5',
        './full_mani_skill_data/OpenCabinetDoor/OpenCabinetDoor_1081_link_0-v0.h5'],
)

rollout_cfg = dict(
    type='BatchRollout',
    with_info=False,
    use_cost=False,
    reward_only=False,
    num_procs=8,
)

eval_cfg = dict(
    type='BatchEvaluation',
    num=100,
    num_procs=2,
    use_hidden_state=False,
    start_state=None,
    save_traj=False,
    save_video=False,
    use_log=True,
)
