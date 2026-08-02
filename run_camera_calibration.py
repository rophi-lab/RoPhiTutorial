"""Run differentiable robot-base → camera extrinsic calibration."""

import argparse
import sys

from omegaconf import OmegaConf

from utils.calibration.diff_base_to_cam import CalibInterrupted, DiffBaseToCamCalibrator


def main():
    parser = argparse.ArgumentParser(
        description="Differentiable base–camera calibration (tutorial 14)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/calibration_tutorial/calib_diff_render.yaml",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))
    calibrator = DiffBaseToCamCalibrator(cfg)
    try:
        calibrator.run()
    except CalibInterrupted:
        print("[Calib] Stopped by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
