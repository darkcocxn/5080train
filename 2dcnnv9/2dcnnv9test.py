from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_MODULE_NAME = "surmod_v8_test_base_for_v9"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_base_module() -> ModuleType:
    return importlib.import_module(BASE_MODULE_NAME)


def configure_v9_test(base: ModuleType) -> None:
    cfg = base.Config
    cfg.MODEL_ROOT_DIR = PROJECT_ROOT / "output" / "2dcnnv9"
    cfg.MODEL_DIR = cfg.MODEL_ROOT_DIR
    cfg.MODEL_WEIGHTS_PATH = cfg.MODEL_DIR / "best_2dcnn_model.pth"

    raw_post_process = os.environ.get("SURMOD_2DCNN_POST_PROCESS_FACTOR")
    if raw_post_process:
        cfg.POST_PROCESS_FACTOR = float(raw_post_process)


def main() -> None:
    base = load_base_module()
    configure_v9_test(base)

    print("\n" + "=" * 60)
    print("2D-CNN Multimodal Model Testing Script v9 (Floors 3 to 7)")
    print("=" * 60 + "\n")

    base.evaluate()

    if base.Config.SCALE_TARGET:
        print("\nHint: the predicted values are restored to drift ratio by dividing the raw output by 1000.")
    if os.environ.get("SURMOD_2DCNN_POST_PROCESS_FACTOR"):
        print(
            "Hint: POST_PROCESS_FACTOR was provided from "
            "SURMOD_2DCNN_POST_PROCESS_FACTOR."
        )


if __name__ == "__main__":
    main()
