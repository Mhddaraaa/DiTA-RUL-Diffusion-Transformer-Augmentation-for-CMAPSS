import argparse
import shlex
import subprocess
import sys
from pathlib import Path


VALID_FDS = ("FD001", "FD002", "FD003", "FD004")
TRAINING_STEPS = ("preprocess", "dte", "diffusion", "rul")


def normalize_fds(fds):
    normalized = [fd.upper() for fd in fds]
    invalid = [fd for fd in normalized if fd not in VALID_FDS]
    if invalid:
        raise ValueError(f"Unsupported FD subset(s): {', '.join(invalid)}")
    return normalized


def add_option(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def add_flag(cmd, flag, enabled):
    if enabled:
        cmd.append(flag)


def printable_command(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(cmd, dry_run=False):
    print("\n$ " + printable_command(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def selected_training_steps(start_at, stop_after):
    start_idx = TRAINING_STEPS.index(start_at)
    stop_idx = TRAINING_STEPS.index(stop_after)
    if start_idx > stop_idx:
        raise ValueError("--start-at must come before or equal --stop-after.")
    return TRAINING_STEPS[start_idx:stop_idx + 1]


def resolved_epochs(args, name):
    specific = getattr(args, f"{name}_epochs")
    return specific if specific is not None else args.epochs


def build_commands(args):
    root = Path(__file__).resolve().parent
    python = sys.executable
    fds = normalize_fds(args.fds)
    output_dir = Path(args.output_dir)
    preprocessed_dir = output_dir / "preprocessed"
    commands = []

    if "preprocess" in args.steps:
        commands.append([
            python,
            str(root / "preprocess_cmapss.py"),
            "--fds",
            *fds,
            "--data-dir",
            args.data_dir,
            "--out-dir",
            str(preprocessed_dir),
            "--window-size",
            str(args.window_size),
            "--step",
            str(args.step),
        ])

    if "dte" in args.steps:
        cmd = [
            python,
            str(root / "DTE_main.py"),
            "--fds",
            *fds,
            "--output-dir",
            str(output_dir),
            "--preprocessed-dir",
            str(preprocessed_dir),
            "--window-size",
            str(args.window_size),
            "--step",
            str(args.step),
            "--valid-size",
            str(args.dte_valid_size),
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
        ]
        add_option(cmd, "--epochs", resolved_epochs(args, "dte"))
        add_option(cmd, "--batch-size", args.batch_size)
        add_option(cmd, "--lr", args.lr)
        add_flag(cmd, "--shuffle-split", args.shuffle_split)
        commands.append(cmd)

    if "diffusion" in args.steps:
        cmd = [
            python,
            str(root / "train_diffusion.py"),
            "--fds",
            *fds,
            "--output-dir",
            str(output_dir),
            "--preprocessed-dir",
            str(preprocessed_dir),
            "--window-size",
            str(args.window_size),
            "--step",
            str(args.step),
            "--valid-fraction",
            str(args.diffusion_valid_fraction),
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
        ]
        add_option(cmd, "--epochs", resolved_epochs(args, "diffusion"))
        add_option(cmd, "--batch-size", args.batch_size)
        add_option(cmd, "--lr", args.lr)
        commands.append(cmd)

    if "rul" in args.steps:
        cmd = [
            python,
            str(root / "train_rul_models.py"),
            "--fds",
            *fds,
            "--output-dir",
            str(output_dir),
            "--window-size",
            str(args.window_size),
            "--step",
            str(args.step),
            "--eval-mode",
            args.rul_eval_mode,
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
            "--gen-batch-size",
            str(args.gen_batch_size),
        ]
        add_option(cmd, "--epochs", resolved_epochs(args, "rul"))
        add_option(cmd, "--batch-size", args.batch_size)
        add_option(cmd, "--lr", args.lr)
        add_option(cmd, "--grad-clip", args.grad_clip)
        add_option(cmd, "--rul-cap", args.rul_cap)
        add_flag(cmd, "--clip-gen", args.clip_gen)
        add_flag(cmd, "--use-huber", args.use_huber)
        commands.append(cmd)

    if args.plots:
        suffix = "_".join(fds)
        dte_plot = output_dir / f"all_z_DTE_{suffix}.png"

        cmd = [
            python,
            str(root / "plot_DTE_all.py"),
            "--fds",
            *fds,
            "--output-dir",
            str(output_dir),
            "--preprocessed-dir",
            str(preprocessed_dir),
            "--window-size",
            str(args.window_size),
            "--step",
            str(args.step),
            "--output",
            str(dte_plot),
        ]
        if args.plot_units is not None:
            cmd.extend(["--units", *[str(unit) for unit in args.plot_units]])
        commands.append(cmd)

        cmd = [
            python,
            str(root / "plot_unit_denoising_generated.py"),
            "--fds",
            *fds,
            "--output-dir",
            str(output_dir),
            "--preprocessed-dir",
            str(preprocessed_dir),
            "--plot-dir",
            str(output_dir),
            "--window-size",
            str(args.window_size),
            "--step",
            str(args.step),
            "--split",
            args.plot_split,
            "--t-fixed",
            str(args.plot_timestep),
        ]
        if args.plot_units is not None:
            cmd.extend(["--units", *[str(unit) for unit in args.plot_units]])
        commands.append(cmd)

    return commands


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the CMAPSS preprocessing, DTE, diffusion, and RUL training pipeline."
    )
    parser.add_argument("--fds", nargs="+", default=["FD001", "FD002"], help="FD subsets to run.")
    parser.add_argument("--data-dir", default="./CMAPSSData", help="Directory containing raw CMAPSS text files.")
    parser.add_argument("--output-dir", default="./output", help="Directory for all generated outputs.")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--start-at", choices=TRAINING_STEPS, default="preprocess")
    parser.add_argument("--stop-after", choices=TRAINING_STEPS, default="rul")
    parser.add_argument("--epochs", type=int, default=None, help="Set DTE, diffusion, and RUL epochs together.")
    parser.add_argument("--dte-epochs", type=int, default=None)
    parser.add_argument("--diffusion-epochs", type=int, default=None)
    parser.add_argument("--rul-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dte-valid-size", type=float, default=0.2)
    parser.add_argument("--diffusion-valid-fraction", type=float, default=0.1)
    parser.add_argument("--shuffle-split", action="store_true", help="Shuffle DTE train/validation split.")
    parser.add_argument("--rul-eval-mode", choices=["window", "engine"], default="engine")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--gen-batch-size", type=int, default=64)
    parser.add_argument("--rul-cap", type=float, default=None)
    parser.add_argument("--clip-gen", action="store_true")
    parser.add_argument("--use-huber", action="store_true")
    parser.add_argument("--plots", action="store_true", help="Run both plotting scripts after training steps.")
    parser.add_argument("--plot-units", nargs="+", type=int, default=None, help="Unit numbers, one per FD.")
    parser.add_argument("--plot-split", default="train", choices=["train", "test"])
    parser.add_argument("--plot-timestep", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.steps = selected_training_steps(args.start_at, args.stop_after)

    commands = build_commands(args)
    for cmd in commands:
        run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
