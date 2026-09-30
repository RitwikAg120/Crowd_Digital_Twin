"""
Check the GB10 (or any machine) is ready before scripts/run_gb10.sh runs:
the code imports, the GPU really runs kernels, the detector weights are in
place, and which datasets are there. Prints a readable report, and with
--env writes KEY=value lines that run_gb10.sh reads.

    python scripts/preflight.py
"""

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_OFFLINE", "1")


def main():
    env, problems = {}, []

    # 1. The code imports
    try:
        import main as _main                              # noqa: F401
        import evaluate as _evaluate                      # noqa: F401
        print("ok   code imports (main.py, evaluate.py)")
        env["CODE_OK"] = 1
    except Exception:
        print("FAIL code does not import:")
        traceback.print_exc()
        env["CODE_OK"] = 0
        problems.append("code import — see the traceback above (missing package? run gb10_setup.sh)")

    # 2. PyTorch and the GPU — a real convolution, not just is_available()
    env["DEVICE"] = "cpu"
    try:
        import torch
        print(f"ok   PyTorch {torch.__version__}, CUDA build {torch.version.cuda}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            try:
                x = torch.randn(1, 3, 64, 64, device="cuda")
                w = torch.randn(8, 3, 3, 3, device="cuda")
                y = torch.nn.functional.conv2d(x, w).half().float().sum().item()
                torch.cuda.synchronize()
                env["DEVICE"] = "cuda:0"
                print(f"ok   GPU {name} (sm_{cap[0]}{cap[1]}) runs kernels")
            except Exception as e:
                print(f"FAIL GPU {name} (sm_{cap[0]}{cap[1]}) is visible but can't run kernels: {e}")
                print(f"     this PyTorch was built for {torch.cuda.get_arch_list()}; install a build "
                      f"that includes sm_{cap[0]}{cap[1]} (see TORCH_INDEX in gb10_setup.sh)")
                problems.append("GPU kernels — falling back to CPU (slow)")
        else:
            print("FAIL PyTorch sees no GPU — steps will run on the CPU (slow)")
            problems.append("no GPU")
    except ImportError:
        print("FAIL PyTorch is not installed (run scripts/gb10_setup.sh)")
        problems.append("PyTorch missing")

    # 3. Detector weights (not in git — copy them from the laptop)
    try:
        from main import Config
        need = [Config.YOLO_MODEL, Config.HBOX_MODEL]
    except Exception:
        need = ["weights/yolo26strained.pt", "weights/yoloheadv26s.pt"]
    missing = [w for w in need if not Path(w).exists()]
    env["WEIGHTS_OK"] = int(not missing)
    if missing:
        print(f"FAIL detector weights missing: {', '.join(missing)}")
        print("     they are not in git; copy them from the laptop, e.g.\n"
              "     scp weights/yolo26strained.pt weights/yoloheadv26s.pt <gb10>:<repo>/weights/")
        problems.append("weights missing — detector steps (bench, mot17, dense) are skipped")
    else:
        print(f"ok   detector weights: {', '.join(need)}")

    # 4. Datasets
    mot = sorted(p.name for p in Path("dataset/MOT17/train").glob("MOT17-*") if (p / "gt/gt.txt").exists()) \
        if Path("dataset/MOT17/train").exists() else []
    env["MOT17_SEQS"] = " ".join(mot)
    print(f"{'ok  ' if mot else 'none'} MOT17 sequences: {', '.join(mot) or '— run scripts/fetch_datasets.py mot17'}")
    ch = Path("dataset/CrowdHuman")
    has_ch = ch.exists() and any(ch.rglob("annotation_train.odgt")) and any(ch.rglob("annotation_val.odgt"))
    env["CROWDHUMAN_OK"] = int(has_ch)
    print(f"{'ok  ' if has_ch else 'none'} CrowdHuman: "
          f"{'annotations found' if has_ch else 'dataset/CrowdHuman with annotation_{train,val}.odgt not found'}")
    jhu = Path("dataset/JHU-Crowd")
    has_jhu = jhu.exists() and any((p / "images").is_dir() for p in jhu.rglob("train"))
    env["JHU_OK"] = int(has_jhu)
    print(f"{'ok  ' if has_jhu else 'none'} JHU-Crowd++: "
          f"{'train split found' if has_jhu else 'dataset/JHU-Crowd not found (optional: dense fine-tune)'}")
    kp = Path("dataset/kumbh_points/train")
    env["KUMBH_POINTS_OK"] = int(kp.exists() and any(kp.glob("*.txt")))
    vids = sorted(p.name for p in Path("videos").glob("*.mp4")) if Path("videos").exists() else []
    env["VIDEOS"] = " ".join(vids)
    print(f"{'ok  ' if vids else 'none'} videos: {', '.join(vids) or '— copy some into videos/ for the benchmarks'}")

    print("\nReady." if not problems else "\nProblems:\n  - " + "\n  - ".join(problems))
    if "--env" in sys.argv:
        with open(sys.argv[sys.argv.index("--env") + 1], "w") as fh:
            for k, v in env.items():
                fh.write(f'{k}="{v}"\n')


if __name__ == "__main__":
    main()
