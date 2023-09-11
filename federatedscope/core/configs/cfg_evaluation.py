from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_evaluation_cfg(cfg):

    # ---------------------------------------------------------------------- #
    # Evaluation related options
    # ---------------------------------------------------------------------- #
    cfg.eval = CN(
        new_allowed=True)  # allow user to add their settings under `cfg.eval`

    cfg.eval.freq = 1
    cfg.eval.metrics = []
    cfg.eval.split = ['test', 'val']
    cfg.eval.report = ['weighted_avg', 'avg', 'fairness',
                       'raw']  # by default, we report comprehensive results
    cfg.eval.best_res_update_round_wise_key = "val_loss"

    # Monitoring, e.g., 'dissim' for B-local dissimilarity
    cfg.eval.monitoring = []
    cfg.eval.count_flops = True

    # ---------------------------------------------------------------------- #
    # wandb related options
    # ---------------------------------------------------------------------- #
    cfg.wandb = CN()
    cfg.wandb.use = False
    cfg.wandb.name_user = ''
    cfg.wandb.name_project = ''
    cfg.wandb.online_track = True
    cfg.wandb.client_train_info = False

    # ---------------------------------------------------------------------- #
    # efficiency related options # This works only for FS-LLM temporarily.
    # ---------------------------------------------------------------------- #

    cfg.eval.efficiency = CN()
    cfg.eval.efficiency.use = False
    cfg.eval.efficiency.freq = 1

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_evaluation_cfg)


def assert_evaluation_cfg(cfg):
    pass


register_config("eval", extend_evaluation_cfg)
