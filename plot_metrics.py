import argparse
import csv
import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def load_metrics(csv_path: str) -> List[Dict[str, Any]]:
    """Load metrics.csv into a list of dicts, converting numeric fields when possible."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"metrics.csv not found: {csv_path}")

    rows: List[Dict[str, Any]] = []
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric-looking fields to float, keep empty as None
            converted = {}
            for k, v in row.items():
                if v is None or v == "":
                    converted[k] = None
                    continue
                try:
                    converted[k] = float(v)
                except ValueError:
                    converted[k] = v
            rows.append(converted)
    return rows


def plot_curves(rows: List[Dict[str, Any]], output_dir: str, show: bool = True) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Separate phases
    train_rows = [r for r in rows if r.get("phase") == "train"]
    val_rows = [r for r in rows if r.get("phase") == "val"]
    test_rows = [r for r in rows if r.get("phase") == "test"]

    # Helper to extract x(epoch) and y(metric)
    def get_xy(rs, metric: str):
        xs, ys = [], []
        for r in rs:
            e = r.get("epoch")
            v = r.get(metric)
            if e is None or v is None:
                continue
            xs.append(int(e))
            ys.append(float(v))
        return xs, ys

    # 1) Train losses
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x_loss, y_loss = get_xy(train_rows, "Train_Loss_epoch")
    x_bal, y_bal = get_xy(train_rows, "Balance_epoch")

    if x_loss and y_loss:
        ax1.plot(x_loss, y_loss, label="Train_Loss_epoch", color="tab:blue")
    if x_bal and y_bal:
        ax1.plot(x_bal, y_bal, label="Balance_epoch", color="tab:orange")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Curves")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig.tight_layout()
    out_path = os.path.join(output_dir, "train_losses.png")
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)

    # 2) Validation metrics：分别绘制 PSNR / SSIM / LPIPS
    x_psnr, y_psnr = get_xy(val_rows, "val_psnr")
    x_ssim, y_ssim = get_xy(val_rows, "val_ssim")
    x_lpips, y_lpips = get_xy(val_rows, "val_lpips")

    # val_psnr
    if x_psnr and y_psnr:
        fig_psnr, ax_psnr = plt.subplots(figsize=(8, 5))
        ax_psnr.plot(x_psnr, y_psnr, label="val_psnr", color="tab:green")
        ax_psnr.set_xlabel("Epoch")
        ax_psnr.set_ylabel("PSNR")
        ax_psnr.set_title("Validation PSNR")
        ax_psnr.grid(True, alpha=0.3)
        ax_psnr.legend()
        fig_psnr.tight_layout()
        out_path = os.path.join(output_dir, "val_psnr.png")
        fig_psnr.savefig(out_path)
        if show:
            plt.show()
        plt.close(fig_psnr)

    # val_ssim
    if x_ssim and y_ssim:
        fig_ssim, ax_ssim = plt.subplots(figsize=(8, 5))
        ax_ssim.plot(x_ssim, y_ssim, label="val_ssim", color="tab:red")
        ax_ssim.set_xlabel("Epoch")
        ax_ssim.set_ylabel("SSIM")
        ax_ssim.set_title("Validation SSIM")
        ax_ssim.grid(True, alpha=0.3)
        ax_ssim.legend()
        fig_ssim.tight_layout()
        out_path = os.path.join(output_dir, "val_ssim.png")
        fig_ssim.savefig(out_path)
        if show:
            plt.show()
        plt.close(fig_ssim)

    # val_lpips
    if x_lpips and y_lpips:
        fig_lpips, ax_lpips = plt.subplots(figsize=(8, 5))
        ax_lpips.plot(x_lpips, y_lpips, label="val_lpips", color="tab:purple")
        ax_lpips.set_xlabel("Epoch")
        ax_lpips.set_ylabel("LPIPS")
        ax_lpips.set_title("Validation LPIPS")
        ax_lpips.grid(True, alpha=0.3)
        ax_lpips.legend()
        fig_lpips.tight_layout()
        out_path = os.path.join(output_dir, "val_lpips.png")
        fig_lpips.savefig(out_path)
        if show:
            plt.show()
        plt.close(fig_lpips)

    # 3) Test metrics (optional, only if present)
    if test_rows:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        x_psnr_t, y_psnr_t = get_xy(test_rows, "test_psnr")
        x_ssim_t, y_ssim_t = get_xy(test_rows, "test_ssim")
        x_lpips_t, y_lpips_t = get_xy(test_rows, "test_lpips")

        if x_psnr_t and y_psnr_t:
            ax1.plot(x_psnr_t, y_psnr_t, label="test_psnr", color="tab:green")
        if x_ssim_t and y_ssim_t:
            ax1.plot(x_ssim_t, y_ssim_t, label="test_ssim", color="tab:red")
        if x_lpips_t and y_lpips_t:
            ax1.plot(x_lpips_t, y_lpips_t, label="test_lpips", color="tab:purple")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Metric value")
        ax1.set_title("Test Metrics")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        fig.tight_layout()
        out_path = os.path.join(output_dir, "test_metrics.png")
        fig.savefig(out_path)
        if show:
            plt.show()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training/validation/test metrics from metrics.csv")
    parser.add_argument(
        "--csv",
        type=str,
        default="/media/wsqlab/more/lqj/moceir/moceir/experiment/2026_01_22_20_18_57/metrics.csv",
        help="Path to metrics.csv (default: /media/wsqlab/more/lqj/moceir/moceir/experiment/2026_01_22_20_18_57/metrics.csv)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save plots (default: same directory as csv)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures interactively, only save to files.",
    )

    args = parser.parse_args()

    csv_path = args.csv
    rows = load_metrics(csv_path)

    if args.outdir is not None:
        outdir = args.outdir
    else:
        outdir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "plots")

    plot_curves(rows, outdir, show=not args.no_show)
    print(f"Plots saved under: {outdir}")


if __name__ == "__main__":
    main()
