"""
简化的配置加载模块 - 直接从 config.py 读取配置
不再需要命令行参数，所有设置都在 config.py 中
"""
import argparse
import os
from pathlib import Path
import config


class ConfigNamespace:
    """简单的配置命名空间类，模拟 argparse.Namespace"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({', '.join(items)})"


def train_options():
    """
    从 config.py 加载所有配置并返回配置对象
    完全移除命令行参数解析，所有配置都在 config.py 中
    """
    model_name = getattr(config, "MODEL", None)
    if not model_name:
        raise ValueError("config.MODEL is required")

    model_cfg = getattr(config, "MODEL_CONFIG", None)
    if model_cfg is None:
        for cand in (
            f"{model_name}_CONFIG",
            f"{str(model_name).upper()}_CONFIG",
        ):
            model_cfg = getattr(config, cand, None)
            if model_cfg is not None:
                break
    
    project_dir = Path(__file__).resolve().parent

    # 合并所有配置到一个命名空间对象
    data_file_dir = Path(str(config.DATA_FILE_DIR)).expanduser()
    if not data_file_dir.is_absolute():
        data_file_dir = (project_dir / data_file_dir).resolve()
    else:
        data_file_dir = data_file_dir.resolve()
    data_file_dir = str(data_file_dir)

    ckpt_dir = Path(str(config.CKPT_DIR)).expanduser()
    if not ckpt_dir.is_absolute():
        ckpt_dir = (project_dir / ckpt_dir).resolve()
    else:
        ckpt_dir = ckpt_dir.resolve()
    ckpt_dir = str(ckpt_dir)

    # 处理外部目录路径
    experiment_dir = Path(str(getattr(config, "EXPERIMENT_DIR", "../experiment"))).expanduser()
    if not experiment_dir.is_absolute():
        experiment_dir = (project_dir / experiment_dir).resolve()
    else:
        experiment_dir = experiment_dir.resolve()
    experiment_dir = str(experiment_dir)

    test_dir = Path(str(getattr(config, "TEST_DIR", "../test"))).expanduser()
    if not test_dir.is_absolute():
        test_dir = (project_dir / test_dir).resolve()
    else:
        test_dir = test_dir.resolve()
    test_dir = str(test_dir)

    results_dir = Path(str(getattr(config, "RESULTS_DIR", "../results"))).expanduser()
    if not results_dir.is_absolute():
        results_dir = (project_dir / results_dir).resolve()
    else:
        results_dir = results_dir.resolve()
    results_dir = str(results_dir)

    # 如果路径不存在，尝试做一次“大小写 Data/data”纠正（常见于 /media 下手工输入）
    if not os.path.exists(data_file_dir):
        alt = None
        if "/media/" in data_file_dir:
            if "/data/" in data_file_dir:
                alt = data_file_dir.replace("/data/", "/Data/")
            elif "/Data/" in data_file_dir:
                alt = data_file_dir.replace("/Data/", "/data/")
        if alt and os.path.exists(alt):
            print(f"[Warn] data_file_dir 不存在：{data_file_dir}，自动改用：{alt}")
            data_file_dir = alt

    base_kwargs = dict(
        # 基础训练设置
        model=config.MODEL,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LR,
        de_type=config.DE_TYPE,
        trainset=config.TRAINSET,
        loss_type=config.LOSS_TYPE,
        patch_size=config.PATCH_SIZE,
        balance_loss_weight=config.BALANCE_LOSS_WEIGHT,
        fft_loss_weight=config.FFT_LOSS_WEIGHT,
        focal_gamma=getattr(config, "FOCAL_GAMMA", 2.0),
        focal_alpha=getattr(config, "FOCAL_ALPHA", 0.1),
        focal_epsilon=getattr(config, "FOCAL_EPSILON", 1e-6),
        de_aux_loss_weight=getattr(config, "DE_AUX_LOSS_WEIGHT", 0.0),
        de_aux_gamma=getattr(config, "DE_AUX_GAMMA", 2.0),
        de_aux_alpha=getattr(config, "DE_AUX_ALPHA", None),
        de_aux_use_external_focal=getattr(config, "DE_AUX_USE_EXTERNAL_FOCAL", False),
        num_workers=config.NUM_WORKERS,
        accum_grad=config.ACCUM_GRAD,
        print_model=getattr(config, "PRINT_MODEL", False),
        check_val_every_n_epoch=getattr(config, "VAL_EVERY_N_EPOCH", 5),
        resume_from=config.RESUME_FROM,
        fine_tune_from=config.FINE_TUNE_FROM,
        checkpoint_id=config.CHECKPOINT_ID,
        benchmarks=config.BENCHMARKS,
        save_results=config.SAVE_RESULTS,
        full_res_eval=getattr(config, "FULL_RES_EVAL", False),

        # 性能相关
        deterministic=getattr(config, "DETERMINISTIC", False),
        benchmark=getattr(config, "BENCHMARK", True),
        precision=getattr(config, "PRECISION", "16-mixed"),
        tf32=getattr(config, "TF32", True),
        log_every_n_steps=getattr(config, "LOG_EVERY_N_STEPS", 50),
        prefetch_factor=getattr(config, "PREFETCH_FACTOR", 2),
        persistent_workers=getattr(config, "PERSISTENT_WORKERS", True),
        
        # 路径设置
        data_file_dir=data_file_dir,
        output_path=config.OUTPUT_PATH,
        wblogger=config.WBLOGGER,
        ckpt_dir=ckpt_dir,
        experiment_dir=experiment_dir,
        test_dir=test_dir,
        results_dir=results_dir,
        num_gpus=config.NUM_GPUS,
    )

    model_kwargs = {}
    if isinstance(model_cfg, dict):
        if model_name == "ACFormer":
            if any(str(k).startswith("ac_") for k in model_cfg.keys()):
                model_kwargs = dict(model_cfg)
            else:
                model_kwargs = dict(
                    ac_dim=model_cfg.get("dim", 48),
                    ac_num_blocks=model_cfg.get("num_blocks", [2, 4, 4, 4]),
                    ac_num_refinement_blocks=model_cfg.get("num_refinement_blocks", 4),
                    ac_channel_heads=model_cfg.get("channel_heads", [1, 2, 4, 8]),
                    ac_spatial_heads=model_cfg.get("spatial_heads", [1, 2, 4, 8]),
                    ac_overlap_ratio=model_cfg.get("overlap_ratio", [0.5, 0.5, 0.5, 0.5]),
                    ac_window_size=model_cfg.get("window_size", 8),
                    ac_spatial_dim_head=model_cfg.get("spatial_dim_head", 16),
                    ac_ffn_expansion_factor=model_cfg.get("ffn_expansion_factor", 2.66),
                    ac_bias=model_cfg.get("bias", False),
                    ac_layernorm_type=model_cfg.get("LayerNorm_type", "WithBias"),
                    ac_M=model_cfg.get("M", 13),
                    ac_ca_heads=model_cfg.get("ca_heads", 2),
                    ac_ca_dim=model_cfg.get("ca_dim", 32),
                    ac_window_size_ca=model_cfg.get("window_size_ca", 8),
                    ac_query_ksize=model_cfg.get("query_ksize", [15, 11, 7, 3, 3]),
                    ac_use_ca=model_cfg.get("use_ca", False),
                )
        else:
            model_kwargs = dict(model_cfg)

    options = ConfigNamespace(**base_kwargs, **model_kwargs)
    
    # Adjust batch size if gradient accumulation is used
    if options.accum_grad > 1:
        options.batch_size = options.batch_size // options.accum_grad
    
    return options