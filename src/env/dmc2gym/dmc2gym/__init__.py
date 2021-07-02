import gym
from gym.envs.registration import register


def make(
        domain_name,
        task_name,
        seed=1,
        visualize_reward=True,
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        episode_length=1000,
        is_distracting_cs=None,
        distracting_cs_intensity=None,
        background_dataset_paths=None,
        environment_kwargs=None,
        setting_kwargs=None,
        time_limit=None,
        channels_first=True
):
    if is_distracting_cs:
        env_id = 'dmc_%s_%s-%s-v1' % (domain_name, task_name, 'dcs')
    else:
        env_id = 'dmc_%s_%s-v1' % (domain_name, task_name)

    if from_pixels:
        assert not visualize_reward, 'cannot use visualize reward when learning from pixels'

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip
    time_limit = 1e6

    if not env_id in gym.envs.registry.env_specs:
        task_kwargs = {}
        if seed is not None:
            task_kwargs['random'] = seed
        if time_limit is not None:
            task_kwargs['time_limit'] = time_limit
        register(
            id=env_id,
            entry_point='dmc2gym.wrappers:DMCWrapper',
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                setting_kwargs=setting_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                is_distracting_cs=is_distracting_cs,
                distracting_cs_intensity=distracting_cs_intensity,
                background_dataset_paths=background_dataset_paths,
                channels_first=channels_first,
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)
