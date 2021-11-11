from semi_seg.hooks import create_infonce_hooks, create_sp_infonce_hooks, create_discrete_mi_consistency_hook


def _hook_config_validator(config, is_pretrain):
    if is_pretrain:
        """Do not accept MeanTeacher and Consistency"""
        pass


def create_hook_from_config(model, config, is_pretrain=False):
    data_name = config["Data"]["name"]
    max_epoch = config["Trainer"]["max_epoch"]
    hooks = []
    if "InfonceParams" in config:
        hook = create_infonce_hooks(model=model, data_name=data_name, **config["InfonceParams"])
        hooks.append(hook)
    if "SPInfonceParams" in config:
        info_hook = create_sp_infonce_hooks(
            model=model, data_name=data_name, max_epoch=max_epoch, **config["SPInfonceParams"]
        )
        hooks.append(info_hook)
    if "DiscreteMIConsistencyParams" in config:
        if is_pretrain:
            raise RuntimeError("DiscreteMIConsistencyParams are not supported for pretrain stage")
        mi_hook = create_discrete_mi_consistency_hook(model=model, **config["DiscreteMIConsistencyParams"])
        hooks.append(mi_hook)

    return hooks
