"""按 CAMP/DAC 的划分真实复制 SUES-200 图像，不使用软链接。

从 Sample4Geo 目录运行：
    python sample4geo/dataset/SUES-200/split_datasets.py

默认读取 ./data/SUES-200，可通过 --path 单独指定 SUES-200 根目录。
相对路径以当前工作目录为准。原始根目录中应有 satellite-view/<地点>/ 和
drone_view_512/<地点>/<高度>/；Training、Testing 也生成在这个根目录中。
"""

import argparse
import shutil
from pathlib import Path


ALTITUDES = (150, 200, 250, 300)
SCENES = tuple(f"{index:04d}" for index in range(1, 201))


def load_split(index_file):
    import yaml

    with index_file.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict) or not isinstance(values.get("index"), list):
        raise ValueError(f"划分文件必须包含 index 列表：{index_file}")
    train_scenes = [str(scene).zfill(4) for scene in values["index"]]
    if len(train_scenes) != 120 or len(set(train_scenes)) != 120:
        raise ValueError("index 必须包含 120 个不重复的训练地点")
    unknown = set(train_scenes).difference(SCENES)
    if unknown:
        raise ValueError(f"地点编号必须在 0001–0200 之间：{sorted(unknown)}")
    train_set = set(train_scenes)
    test_scenes = [scene for scene in SCENES if scene not in train_set]
    return train_scenes, test_scenes


def find_view_root(root, candidates):
    matches = [root / name for name in candidates if (root / name).is_dir()]
    if len(matches) != 1:
        raise ValueError(
            f"{root} 下应存在且仅存在一个原始视角目录：{', '.join(candidates)}；"
            f"实际找到：{[path.name for path in matches]}"
        )
    return matches[0]


def prepare_dataset(root, index_file, altitudes):
    root = Path(root).expanduser().resolve()
    index_file = Path(index_file).expanduser().resolve()
    train_scenes, test_scenes = load_split(index_file)
    satellite_root = find_view_root(
        root, ("satellite-view", "satellite_view_512", "satellite-view_512")
    )
    drone_root = find_view_root(
        root, ("drone_view_512", "drone-view", "drone-view_512")
    )

    # 地点划分和六个目录与 CAMP/DAC 一致；gallery 包含全部 200 个地点。
    views = (
        ("Training", "satellite", train_scenes, satellite_root, False),
        ("Training", "drone", train_scenes, drone_root, True),
        ("Testing", "query_satellite", test_scenes, satellite_root, False),
        ("Testing", "query_drone", test_scenes, drone_root, True),
        ("Testing", "gallery_satellite", SCENES, satellite_root, False),
        ("Testing", "gallery_drone", SCENES, drone_root, True),
    )
    jobs = []
    for altitude in altitudes:
        for split, view, scenes, source_root, has_altitude in views:
            for scene in scenes:
                source = source_root / scene
                if has_altitude:
                    source /= str(altitude)
                destination = root / split / str(altitude) / view / scene
                jobs.append((source, destination))

    # 全部检查完成后才开始复制，避免缺少某个高度时先产生部分副本。
    for source, destination in jobs:
        if not source.is_dir() or not any(path.is_file() for path in source.iterdir()):
            raise FileNotFoundError(f"原始图像目录不存在或为空：{source}")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"目标已存在，不会覆盖：{destination}。若数据已经整理完成，可直接训练。"
            )

    print(f"数据根目录：{root}")
    print(f"划分文件：{index_file}")
    print("训练地点：120；测试地点：80；gallery 地点：200")
    print(f"准备高度：{list(altitudes)}；采用真实复制，原始数据保留")
    previous_view = None
    for source, destination in jobs:
        if destination.parent != previous_view:
            previous_view = destination.parent
            print(f"复制 {previous_view.relative_to(root)}", flush=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    print(f"完成：复制了 {len(jobs)} 个地点目录。训练配置的 data_folder 使用 {root}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_path = "./data/SUES-200"
    parser.add_argument(
        "--path", type=Path, default=default_path,
        help=f"原始/整理后的 SUES-200 根目录（默认：{default_path}）",
    )
    parser.add_argument(
        "--index-file", type=Path,
        default=Path(__file__).resolve().with_name("indexs.yaml"),
        help="默认使用本脚本旁的 indexs.yaml，与工作目录无关",
    )
    parser.add_argument(
        "--altitudes", type=int, nargs="+", choices=ALTITUDES, default=list(ALTITUDES),
        help="要准备的高度，默认生成全部四个高度",
    )
    args = parser.parse_args(argv)
    if len(set(args.altitudes)) != len(args.altitudes):
        parser.error("--altitudes 不能重复")
    try:
        prepare_dataset(args.path, args.index_file, args.altitudes)
    except (OSError, ValueError) as error:
        parser.exit(2, f"数据准备失败：{error}\n")


if __name__ == "__main__":
    main()
