import wandb, os

api = wandb.Api()
# 替换为你想要下载的run的路径
run = api.run("idear-fudan-university/X2_Tracking/8a6s5o4c")

# 下载到 logs/rsl_rl/<run_name>/ 目录
save_dir = f"logs/rsl_rl/x2_flat/{run.name}"
os.makedirs(save_dir, exist_ok=True)

targets = ["model_29999.pt", f"{run.name}.onnx", "config.yaml"]
#如果model_29999.pt不存在，则下载最新的checkpoint文件
if not os.path.exists(os.path.join(save_dir, "model_29999.pt")):
    checkpoints = [f for f in run.files() if f.name.startswith("model_") and f.name.endswith(".pt")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda f: int(f.name.split("_")[1].split(".")[0]))
        targets[0] = latest_checkpoint.name
    else:
        print("No checkpoint files found, skipping model download.")
        targets[0] = None


for fname in targets:
    if fname is None:
        continue

    try:
        f = run.file(fname)
        f.download(root=save_dir, replace=True)
        print(f"✓ {fname}")
    except Exception as e:
        print(f"✗ {fname}: {e}")

print(f"\n已保存到: {save_dir}/")