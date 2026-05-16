config = {
    # ---- Data / architecture sizes ----
    "input_size": 25,          # feature_cols = 1..25
    "window_size": 30,

    # ---- DTE (TSHAE) ----
    "hidden_size": 64,
    "d_model": 64,
    "num_heads": 4,
    "latent_dim": 32,
    "num_layers": 2,
    "bidirectional": True,

    "wavelet_kernel_size": (3, 5, 7),

    "regression_dims": 64,
    "reconstruct": True,
    
     # ---- Dropouts ----
    "dropout_lstm_encoder": 0.1,
    "dropout_lstm_decoder": 0.1,
    "dropout_layer_encoder": 0.1,
    "dropout_layer_decoder": 0.1,
    "dropout_regressor": 0.1,
    "dropout_latent": 0.1,     # for DropBlockLatent in TSHAE

    # ---- Training (shared defaults) ----
    "lr": 1e-4,
    "batch_size": 32,
    "max_epochs": 50,
    "output_dir": "./output",
    "data_dir": "./CMAPSSData",

    # ---- DTE Loss weights ----
    "KLLoss_weight": 1.0,
    "RegLoss_weight": 1.0,
    "ReconLoss_weight": 1.0,

    # start lower; raise later if stable
    "TripletLoss_weight": 2.0,
    "TripletLoss_margin": 0.4,
    "TripletLoss_p": 2,

    "DecorLoss_weight": 0.0,

    # ---- Diffusion (DDPM + DiffWave) ----
    "noise_steps": 50,
    "beta_start": 0.0004,
    "beta_end": 0.05,
    "schedule_name": "linear",

    "residual_channels": 64,
    "residual_layers": 4,
    "ff_dim": 256,          # add explicitly (optional but recommended)
    "film_scale": 0.5,      # safer FiLM

    # ---- Diffusion auxiliary losses ----
    "Trend_weight": 0.1,
    "Spec_weight": 0.05,
    "Spec_keep_ratio": 0.2,

    # ---- Stability knobs (highly recommended) ----
    "grad_clip": 1.0,
    "use_amp": True,
    "ema_beta": 0.999,
    "ema_start": 2000,

    # ---- LR scheduler (optional) ----
    "lr_step": 5,
    "lr_gamma": 0.9,
}