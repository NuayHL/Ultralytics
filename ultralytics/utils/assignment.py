from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.mla import (TaskAlignedAssigner_Record,
                                   TaskAlignedAssigner_abtest,
                                   TaskAlignedAssigner_Scale_abtest,
                                   TaskAlignedAssigner_BCE,
                                   TaskAlignedAssigner_BCE1,
                                   TaskAlignedAssigner_BCE2,
                                   TaskAlignedAssigner_Scale,
                                   TaskAlignedAssigner_Scale_BCE1,
                                   TaskAlignedAssigner_Scale_BCE2,
                                   TaskAlignedAssigner_General,
                                   TaskAlignedAssigner_MixAssign,
                                   TaskAlignedAssigner_dynamicK,
                                   TaskAlignedAssigner_Scale_dynamicK)
from ultralytics.utils.mla_hbg import (TaskAlignedAssigner_hbg,
                                       TaskAlignedAssigner_hbg_with_Scale)
from ultralytics.utils.mla_kde import (TaskAlignedAssigner_kde_dynamicK,
                                       TaskAlignedAssigner_kde)
from ultralytics.utils.mla_dab import (TaskAlignedAssigner_dab,
                                       TaskAlignedAssigner_dabsep,
                                       TaskAlignedAssigner_dabsepScore,
                                       TaskAlignedAssigner_dabsepScore1)
from ultralytics.utils.tal import TaskAlignedAssigner


ASSIGN_USE_STRIDE = (TaskAlignedAssigner_Scale,
                     TaskAlignedAssigner_Scale_BCE1,
                     TaskAlignedAssigner_Scale_BCE2,
                     TaskAlignedAssigner_hbg_with_Scale,
                     TaskAlignedAssigner_Scale_dynamicK,
                     TaskAlignedAssigner_Scale_abtest)

# bce1 is a mistake so did not add in it
ASSIGN_USE_LOGIST = (TaskAlignedAssigner_BCE,
                     TaskAlignedAssigner_BCE2,
                     TaskAlignedAssigner_General,
                     TaskAlignedAssigner_Scale_BCE2)

ASSIGN_USE_HBG = (TaskAlignedAssigner_hbg,
                  TaskAlignedAssigner_hbg_with_Scale)

def LOGGER_INFO(assigner_type):
    LOGGER.info(f"{colorstr('Using Logist pd_socre')}: {type(assigner_type) in ASSIGN_USE_LOGIST}")
    LOGGER.info(f"{colorstr('Using Stride pd_socre')}: {type(assigner_type) in ASSIGN_USE_STRIDE}")

def get_task_aligned_assigner(cfg: dict, nc=80, **kwargs):
    assigner_type = cfg.get("assigner_type", "TaskAlignedAssigner")
    assigner = None
    _kwargs = dict(topk=cfg.get("topk", 10),
                   num_classes=nc,
                   alpha=cfg.get("alpha", 0.5),
                   beta=cfg.get("beta", 6.0))

    if assigner_type == "TaskAlignedAssigner":
        assigner = TaskAlignedAssigner(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_BCE":
        assigner = TaskAlignedAssigner_BCE(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_BCE1":
        assigner = TaskAlignedAssigner_BCE1(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_BCE2":
        assigner = TaskAlignedAssigner_BCE2(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_MixAssign":
        assigner = TaskAlignedAssigner_MixAssign(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_abtest":
        _kwargs['score_alpha'] = cfg.get("score_alpha", 0.5)
        _kwargs['score_beta'] = cfg.get("score_beta", 6.0)
        assigner = TaskAlignedAssigner_abtest(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_Scale_abtest":
        _kwargs['scale_ratio'] = cfg.get("scale_ratio", 1.0)
        _kwargs['score_alpha'] = cfg.get("score_alpha", 0.5)
        _kwargs['score_beta'] = cfg.get("score_beta", 6.0)
        assigner = TaskAlignedAssigner_Scale_abtest(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_dab":
        assigner = TaskAlignedAssigner_dab(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_dabsep":
        _kwargs['score_alpha'] = cfg.get("score_alpha", 1.0)
        _kwargs['score_beta'] = cfg.get("score_beta", 4.0)
        assigner = TaskAlignedAssigner_dabsep(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_dabsepScore":
        _kwargs['score_alpha'] = cfg.get("score_alpha", [0.3, 2.5, 2.0, 1.7])
        _kwargs['score_beta'] = cfg.get("score_beta", [0.3, 5.0, 2.0, 1.7])
        assigner = TaskAlignedAssigner_dabsepScore(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_dabsepScore1":
        _kwargs['score_alpha'] = cfg.get("score_alpha", [-0.1, 1.3])
        _kwargs['score_beta'] = cfg.get("score_beta", [0.33, 3.0])
        assigner = TaskAlignedAssigner_dabsepScore1(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_dynamicK":
        _kwargs['min_topk'] = cfg.get("min_topk", 4)
        _kwargs['max_topk'] = cfg.get("max_topk", 10)
        _kwargs['metric_sum_thr'] = cfg.get("metric_sum_thr", 3)
        assigner = TaskAlignedAssigner_dynamicK(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_Scale_dynamicK":
        _kwargs['min_topk'] = cfg.get("min_topk", 4)
        _kwargs['max_topk'] = cfg.get("max_topk", 10)
        _kwargs['metric_sum_thr'] = cfg.get("metric_sum_thr", 3)
        _kwargs['scale_ratio'] = cfg.get("scale_ratio", 1.0)
        assigner = TaskAlignedAssigner_Scale_dynamicK(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_kde":
        _kwargs['kde_max_topk'] = cfg.get('kde_max_topk', 10)
        _kwargs['kde_min_topk'] = cfg.get('kde_min_topk', 3)
        _kwargs['kde_metric_sum_thr'] = cfg.get('kde_metric_sum_thr', 4.0)
        _kwargs['bandwidth_scale_factor'] = cfg.get("bandwidth_scale_factor", 0.15)
        assigner = TaskAlignedAssigner_kde(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_kde_dynamicK":
        _kwargs['min_topk'] = cfg.get("min_topk", 4)
        _kwargs['max_topk'] = cfg.get("max_topk", 10)
        _kwargs['metric_sum_thr'] = cfg.get("metric_sum_thr", 3)
        _kwargs['kde_max_topk'] = cfg.get('kde_max_topk', 10)
        _kwargs['kde_min_topk'] = cfg.get('kde_min_topk', 3)
        _kwargs['kde_metric_sum_thr'] = cfg.get('kde_metric_sum_thr', 4.0)
        _kwargs['bandwidth_scale_factor'] = cfg.get("bandwidth_scale_factor", 0.15)
        assigner = TaskAlignedAssigner_kde_dynamicK(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_hbg":
        _kwargs['hbg_topk'] = cfg.get("hbg_topk", 20)
        assigner = TaskAlignedAssigner_hbg(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_hbg_with_Scale":
        _kwargs['scale_ratio'] = cfg.get("scale_ratio", 1.0)
        assigner = TaskAlignedAssigner_hbg_with_Scale(**_kwargs)

    elif assigner_type == "TaskAlignedAssigner_General":
        _kwargs['align_type'] = cfg.get("align_type", "tal")
        assigner = TaskAlignedAssigner_General(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_Scale":
        _kwargs['scale_ratio'] = cfg.get("scale_ratio", 1.0)
        assigner = TaskAlignedAssigner_Scale(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_Scale_BCE1":
        _kwargs['scale_ratio'] = cfg.get("scale_ratio", 1.0)
        assigner = TaskAlignedAssigner_Scale_BCE1(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_Scale_BCE2":
        _kwargs['scale_ratio'] = cfg.get("scale_ratio", 1.0)
        assigner = TaskAlignedAssigner_Scale_BCE2(**_kwargs)
    elif assigner_type == "TaskAlignedAssigner_Record":
        _kwargs['dir_name'] = cfg.get("dir_name", 'test')
        _kwargs['save_step'] = cfg.get("save_step", 10)
        assigner = TaskAlignedAssigner_Record(**_kwargs)
    else:
        raise ValueError(f"Unknown assigner type: {assigner_type}")

    LOGGER.info(f"\r{colorstr('Using '+ str(type(assigner)))}: {_kwargs}")
    LOGGER_INFO(assigner)
    return assigner

