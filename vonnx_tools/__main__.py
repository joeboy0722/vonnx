import argparse
from .common import TASKS, SCALES, RESOLUTIONS, STREAMS


def main():
    parser = argparse.ArgumentParser(description="vONNX reproducible YOLOv8 suite")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--scales", nargs="+", choices=SCALES, default=list(SCALES))
    p.add_argument("--datasets", choices=["none", "smoke", "full"], default="smoke")
    p.add_argument("--force", action="store_true")
    p = commands.add_parser("validate")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16"], default="auto")
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--scales", nargs="+", choices=SCALES, default=list(SCALES))
    p.add_argument("--sizes", nargs="+", type=int, default=list(RESOLUTIONS))
    p.add_argument("--full", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default="runs/validation")
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--case")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--worker")
    p = commands.add_parser("benchmark")
    p.add_argument("--sizes", nargs="+", type=int, default=list(RESOLUTIONS))
    p.add_argument("--streams", nargs="+", type=int, default=list(STREAMS))
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Maximum measured requests per model/case; 0 means duration-only",
    )
    p.add_argument("--output", default="runs/benchmark")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--case")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--worker")
    p.add_argument("--list", action="store_true")
    p.add_argument("--validation", default="runs/validation")
    p.add_argument(
        "--allow-sampled-validation",
        action="store_true",
        help="Accept successful COCO subset validation; record sample size in benchmark results",
    )
    p.add_argument(
        "--allow-unvalidated", action="store_true", help="Development smoke only; report marked unvalidated"
    )
    p = commands.add_parser("report")
    p.add_argument("--input", default="runs/benchmark")
    p = commands.add_parser("optimize", help="Bounded s/640 optimization; never starts full tests")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument(
        "--phase", choices=["check", "baseline", "quick", "confirm", "diagnostic", "report"], default="quick"
    )
    p.add_argument("--output", default="runs/optimization")
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--case")
    p.add_argument("--worker")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare

        prepare(args)
    elif args.command == "validate":
        from .validate import validate

        validate(args)
    elif args.command == "benchmark":
        from .benchmark import benchmark

        benchmark(args)
    elif args.command == "optimize":
        from .optimize import optimize

        optimize(args)
    else:
        from .report import report

        report(args.input)


if __name__ == "__main__":
    main()
