"""训练 YOLO-FRP 模型的脚本。

该脚本基于 Ultralytics YOLO 接口，默认使用论文中给出的改进
YOLOv10-FRP 配置以及用户提供的数据集路径。
"""

from pathlib import Path
import argparse

from ultralytics import YOLO


DEFAULT_MODEL_CFG = Path("ultralytics/cfg/models/v10/yolov10-frp.yaml")
DEFAULT_DATA_CFG = Path(r"E:\\yolov10-main\\datasets\\data.yaml")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练改进的 YOLO-FRP 模型")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_CFG,
        help="模型配置文件路径，默认为改进后的 YOLO-FRP 配置",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_CFG,
        help="数据集 data.yaml 文件路径",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="训练轮次 (epochs)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="每个批次的样本数量 (batch size)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="训练图像输入尺寸 (imgsz)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="训练所用的计算设备标识，默认使用第一张 GPU",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/train"),
        help="训练结果保存目录",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov10_frp",
        help="实验名称",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="是否从 project/name/weights/last.pt 恢复训练",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader 工作进程数量",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg = args.model.expanduser().resolve()
    data_cfg = args.data

    model = YOLO(str(model_cfg))

    train_kwargs = dict(
        data=str(data_cfg),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project),
        name=args.name,
        resume=args.resume,
        workers=args.workers,
    )

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
