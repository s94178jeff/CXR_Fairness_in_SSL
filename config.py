import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Training configuration parser")

    # ==========================
    # 基本訓練參數
    # ==========================
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay (default: 0.0)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum for optimizer (default: 0.9)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of data loading workers (default: 1)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda', 'cpu', or 'mps' (default: auto-detect)")
    parser.add_argument("--exp", type=str, default="",
                        help="Experiment name")

    # ==========================
    # 資料集與偏差設定
    # ==========================
    parser.add_argument("--dataset", type=str, default="mimic",
                        choices=["mimic", "mimic_ssl", "covid", "covid_ssl"],
                        help="Dataset to train on")
    parser.add_argument("--percent", type=str, default="0.0",
                        help="Percentage of conflict/bias in dataset")
    parser.add_argument("--shortcut_type", type=str, default="no",
                        help="MIMIC shortcut type (default: no)")
    parser.add_argument("--group_type", type=str, default="no",
                        choices=["age", "gender", "race", "no", "LO"],
                        help="Demographic grouping for fairness evaluation")

    # ==========================
    # 訓練控制
    # ==========================
    parser.add_argument("--continue_train", "-c", action="store_true", default=False,
                        help="Continue training from checkpoint")
    parser.add_argument("--local", "-l", action="store_true",
                        help="Run locally (disable wandb, tensorboard)")
    parser.add_argument("--result_root", type=str, default="./result",
                        help="Path for saving results and checkpoints")

    # ==========================
    # 實驗方法 & 模型
    # ==========================
    parser.add_argument("--model", type=str, default="",
                        help="Model name (e.g., resnet18, densenet121)")
    parser.add_argument("--method", type=str, default="vanilla",
                        choices=["aug_vanilla", "vanilla", "fit"],
                        help="Training method")
    parser.add_argument("--shorTest", action="store_true",
                        help="Enable shortcut testing for vanilla CNN")
    parser.add_argument("--use_bias_label", action="store_true",
                        help="Use bias labels for training")

    # ==========================
    # 自監督學習 (SSL) 設定
    # ==========================
    parser.add_argument("--ssl_ckpt_path", type=str, default="",
                        help="Path to SSL encoder checkpoint")
    parser.add_argument("--ssl_type", type=str, default="",
                        choices=["", "simsiam", "byol", "simclr", "dino", "swav"],
                        help="Type of SSL encoder")
    return parser