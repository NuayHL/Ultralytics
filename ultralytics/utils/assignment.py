from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.mla import TaskAlignedAssigner_Record
from ultralytics.utils.tal import TaskAlignedAssigner


def get_task_aligned_assigner(cfg: dict, nc=80, **kwargs):
    assigner_type = cfg.get("assigner_type", "TaskAlignedAssigner")
    if assigner_type == "TaskAlignedAssigner":
        _kwargs = dict(topk = cfg.get("topk", 10),
                     num_classes = nc,
                     alpha = cfg.get("alpha", 0.5),
                     beta = cfg.get("beta", 6.0))

        LOGGER.info(f"\r{colorstr('Using '+assigner_type)}: {_kwargs}")
        return TaskAlignedAssigner(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_Record":
        _kwargs = dict(topk=cfg.get("topk", 10),
                       num_classes=nc,
                       alpha=cfg.get("alpha", 0.5),
                       beta=cfg.get("beta", 6.0),
                       dir_name=cfg.get("dir_name", 'test'),
                       save_step=cfg.get("save_step", 10))

        LOGGER.info(f"\r{colorstr('Using ' + assigner_type)}: {_kwargs}")
        return TaskAlignedAssigner_Record(**_kwargs)

    else:
        raise ValueError(f"Unknown assigner type: {assigner_type}")
