import argparse
import os
import numpy as np


VALID_FDS = ("FD001", "FD002", "FD003", "FD004")


def normalize_fds(fds):
    normalized = [fd.upper() for fd in fds]
    invalid = [fd for fd in normalized if fd not in VALID_FDS]
    if invalid:
        raise ValueError(f"Unsupported FD subset(s): {', '.join(invalid)}")
    return normalized


def parse_feature_cols(value):
    if value is None:
        return None

    cols = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cols.extend(range(int(start), int(end) + 1))
        else:
            cols.append(int(part))
    return np.asarray(cols, dtype=np.int64)


class CMAPSSPreprocessor:
    """
    Build sliding windows and aligned alpha (per-unit progress) for CMAPSS FD001-004.

    Saves:
      preprocessed_data_{prefix}.npy   : [N, L, F]
      preprocessed_ruls_{prefix}.npy   : [N]
      preprocessed_alpha_{prefix}.npy  : [N] in [0,1]
      preprocessed_units_{prefix}.npy  : [N]
      scaler_min_{fd}.npy, scaler_max_{fd}.npy (train only)
    """

    def __init__(self, data_dir, out_dir="output/preprocessed"):
        self.data_dir = data_dir
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def _load_raw(self, fd):
        train_path = os.path.join(self.data_dir, f"train_{fd}.txt")
        test_path  = os.path.join(self.data_dir, f"test_{fd}.txt")
        rul_path   = os.path.join(self.data_dir, f"RUL_{fd}.txt")

        train_raw = np.loadtxt(train_path, dtype=np.float32)
        test_raw  = np.loadtxt(test_path,  dtype=np.float32)
        rul_end   = np.loadtxt(rul_path,   dtype=np.float32)
        return train_raw, test_raw, rul_end

    def _compute_train_rul_per_row(self, unit_ids):
        # RUL decreases from (T-1) to 0 within each unit
        rul = np.zeros(len(unit_ids), dtype=np.float32)
        for u in np.unique(unit_ids):
            idx = np.where(unit_ids == u)[0]
            L = len(idx)
            # Cycle positions run from 0 to L-1, so RUL is L-1 minus the position.
            rul[idx] = (L - 1) - np.arange(L, dtype=np.float32)
        return rul

    def _compute_test_rul_per_row(self, unit_ids, rul_end):
        # test file is truncated; final RUL at end of observed sequence is given by rul_end per unit
        rul = np.zeros(len(unit_ids), dtype=np.float32)
        unique_units = np.unique(unit_ids).astype(int)
        assert len(unique_units) == len(rul_end), "RUL_end length mismatch with # test units"

        for j, u in enumerate(unique_units):
            idx = np.where(unit_ids == u)[0]
            L = len(idx)
            # The observed end has rul_end[j]; earlier rows have larger RUL.
            rul[idx] = rul_end[j] + (L - 1 - np.arange(L, dtype=np.float32))
        return rul

    def _minmax_fit(self, train_feat):
        feat_min = train_feat.min(axis=0)
        feat_max = train_feat.max(axis=0)
        denom = feat_max - feat_min
        denom[denom == 0] = 1.0
        return feat_min, feat_max, denom

    def _minmax_transform(self, feat, feat_min, denom):
        return (feat - feat_min) / denom

    def _create_windows(self, feat_norm, rul_per_row, unit_ids, window_size, step):
        windows, ruls, alphas, units = [], [], [], []

        for u in np.unique(unit_ids):
            idx_u = np.where(unit_ids == u)[0]
            T = len(idx_u)
            if T < window_size:
                continue

            # start positions in local unit coordinates
            for start in range(0, T - window_size + 1, step):
                w_idx = idx_u[start:start + window_size]
                windows.append(feat_norm[w_idx])             # [L,F]
                ruls.append(rul_per_row[w_idx[-1]])          # RUL at window end
                units.append(int(u))

                # Alpha is the normalized cycle position at the end of the window.
                end_pos = start + window_size - 1
                alpha = end_pos / (T - 1 + 1e-12)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                alphas.append(alpha)

        windows = np.stack(windows).astype(np.float32)      # [N,L,F]
        ruls    = np.array(ruls, dtype=np.float32)          # [N]
        alphas  = np.array(alphas, dtype=np.float32)        # [N]
        units   = np.array(units, dtype=np.int32)           # [N]
        return windows, ruls, alphas, units

    def preprocess_fd(self, fd="FD001", window_size=30, step=1, feature_cols=None):
        train_raw, test_raw, rul_end = self._load_raw(fd)

        # Feature selection: columns 1..25 are used by the model.
        if feature_cols is None:
            feature_cols = np.arange(1, 26)

        train_units = train_raw[:, 0].astype(int)
        test_units  = test_raw[:, 0].astype(int)

        train_feat = train_raw[:, feature_cols]
        test_feat  = test_raw[:, feature_cols]

        # RUL per row
        train_rul_row = self._compute_train_rul_per_row(train_units)
        test_rul_row  = self._compute_test_rul_per_row(test_units, rul_end)

        # Fit the scaler on the training split only.
        feat_min, feat_max, denom = self._minmax_fit(train_feat)
        train_norm = self._minmax_transform(train_feat, feat_min, denom)
        test_norm  = self._minmax_transform(test_feat,  feat_min, denom)

        # Create windows
        train_X, train_y, train_a, train_u = self._create_windows(
            train_norm, train_rul_row, train_units, window_size, step
        )
        test_X, test_y, test_a, test_u = self._create_windows(
            test_norm, test_rul_row, test_units, window_size, step
        )

        # Save
        prefix_train = f"{fd}_train_ws{window_size}_step{step}"
        prefix_test  = f"{fd}_test_ws{window_size}_step{step}"

        np.save(os.path.join(self.out_dir, f"preprocessed_data_{prefix_train}.npy"), train_X)
        np.save(os.path.join(self.out_dir, f"preprocessed_ruls_{prefix_train}.npy"), train_y)
        np.save(os.path.join(self.out_dir, f"preprocessed_alpha_{prefix_train}.npy"), train_a)
        np.save(os.path.join(self.out_dir, f"preprocessed_units_{prefix_train}.npy"), train_u)

        np.save(os.path.join(self.out_dir, f"preprocessed_data_{prefix_test}.npy"), test_X)
        np.save(os.path.join(self.out_dir, f"preprocessed_ruls_{prefix_test}.npy"), test_y)
        np.save(os.path.join(self.out_dir, f"preprocessed_alpha_{prefix_test}.npy"), test_a)
        np.save(os.path.join(self.out_dir, f"preprocessed_units_{prefix_test}.npy"), test_u)

        np.save(os.path.join(self.out_dir, f"scaler_min_{fd}.npy"), feat_min.astype(np.float32))
        np.save(os.path.join(self.out_dir, f"scaler_max_{fd}.npy"), feat_max.astype(np.float32))

        print(f"[{fd}] train windows: {train_X.shape}, test windows: {test_X.shape}")
        return (train_X, train_y, train_a, train_u), (test_X, test_y, test_a, test_u)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess CMAPSS FD subsets into normalized sliding-window arrays."
    )
    parser.add_argument("--fds", nargs="+", default=list(VALID_FDS), help="FD subsets to preprocess.")
    parser.add_argument("--data-dir", default="./CMAPSSData", help="Directory containing raw CMAPSS text files.")
    parser.add_argument("--out-dir", default="output/preprocessed", help="Directory for generated .npy files.")
    parser.add_argument("--window-size", type=int, default=30, help="Sliding-window length.")
    parser.add_argument("--step", type=int, default=1, help="Stride between consecutive windows.")
    parser.add_argument(
        "--feature-cols",
        default=None,
        help="Raw zero-based columns to keep, such as '1-25' or '1,2,5'. Defaults to 1-25.",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    fds = normalize_fds(args.fds)
    feature_cols = parse_feature_cols(args.feature_cols)

    pp = CMAPSSPreprocessor(data_dir=args.data_dir, out_dir=args.out_dir)
    for fd in fds:
        pp.preprocess_fd(
            fd=fd,
            window_size=args.window_size,
            step=args.step,
            feature_cols=feature_cols,
        )


if __name__ == "__main__":
    main()
