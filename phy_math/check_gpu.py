import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        backend = "ROCm" if torch.version.hip else "CUDA"
        gpu_name = torch.cuda.get_device_name(device)
        print(f" -> Training device: GPU ({backend}) - {gpu_name}")
    else:
        print(" -> Training device: CPU")


if __name__ == "__main__":
    main()
