import os
import sys
from pathlib import Path

import torch
import pandas as pd


sys.path.append(f"{Path(__file__).resolve().parent}")


def check_speaker_dir() -> list:
    msg = ["空(随机)"]

    current_dir = f"{Path(__file__).resolve().parent}"
    if not os.path.exists(f"{current_dir}/sampled_speaker"):
        os.makedirs(f"{current_dir}/sampled_speaker")

    if f"{current_dir}/sampled_speaker" not in sys.path:
        sys.path.append(f"{current_dir}/sampled_speaker")

    files = [file for file in os.listdir(f"{current_dir}/sampled_speaker") if file.endswith("csv")]
    msg.extend(list(map(lambda x: Path(x).resolve().stem, files)))
    return msg


def generate_speaker_tensor(mean: float = 0.0, std: float = 15.247) -> torch.Tensor:
    return torch.normal(mean, std, size=(768,))


def generate_speaker_tensor_a() -> torch.Tensor:
    std, mean = torch.load(f'{Path(__file__).resolve().parent}/models/asset/spk_stat.pt').chunk(2)
    rand_spk = torch.randn(768) * std + mean
    return rand_spk


def save_speaker_tensor_to_csv(speaker_name: str, tensor: torch.Tensor) -> str:
    msg = "succeed"
    speaker_path = f"{Path(__file__).resolve().parent}/sampled_speaker"
    try:
        df = pd.DataFrame({"speaker": [float(i) for i in tensor]})
        df.to_csv(f"{speaker_path}/{speaker_name}.csv", index=False, header=False)
    except Exception as e:
        print(f"存储 speaker_tensor 时发生错误：{e}")
        msg = "fail"
    finally:
        return msg


def load_speaker_tensor_from_csv(speaker_name: str) -> torch.Tensor:
    speaker_path = f"{Path(__file__).resolve().parent}/sampled_speaker"
    d_s = pd.read_csv(f"{speaker_path}/{speaker_name}.csv", header=None).iloc[:, 0]
    _speaker_tensor = torch.tensor(d_s.values)
    return _speaker_tensor


if __name__ in "__main__":
    # y = check_speaker_dir()
    # print(y)
    # speaker_tensor = generate_speaker_tensor()
    # speaker_tensor = generate_speaker_tensor_a()
    # print(speaker_tensor)
    # sm = save_speaker_tensor_to_csv("niuniu", speaker_tensor)
    # print(sm)
    # lm = load_speaker_tensor_from_csv("niuniu")
    # print(lm)
    pass
