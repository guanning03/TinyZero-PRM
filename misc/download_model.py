import os
import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace model to specified location")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="HuggingFace model ID (e.g., guanning-ai/maze_sft_weights_1207)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory. Default: $CACHE/hf_models/<model_name>")
    parser.add_argument("--cache", "-c", type=str, default=None,
                        help="Cache directory. Default: $CACHE or ~/.cache")
    args = parser.parse_args()

    model_name = args.model

    # 1) 解析并准备 cache 目录
    CACHE = args.cache or os.path.expanduser(os.environ.get("CACHE", "~/.cache"))
    assert CACHE, "Cache directory is empty. Please set --cache or $CACHE environment variable."

    # 2) 把 HF 的所有缓存也指向 CACHE，避免使用 ~/.cache
    hf_home = os.path.join(CACHE, "hf_home")
    hf_hub_cache = os.path.join(hf_home, "hub")
    tfm_cache = os.path.join(hf_home, "transformers")
    os.makedirs(hf_hub_cache, exist_ok=True)
    os.makedirs(tfm_cache, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hf_hub_cache
    os.environ["TRANSFORMERS_CACHE"] = tfm_cache

    # 3) 最终落盘目录（你想要的目标目录）
    local_path = args.output or os.path.join(CACHE, f"hf_models/{model_name}")
    local_path = os.path.expanduser(local_path)
    os.makedirs(local_path, exist_ok=True)

    print(f"Downloading repo {model_name} to {local_path} ...")
    # 4) 直接把仓库文件下载到 local_path；
    #    同时指定 cache_dir 也在 CACHE 下，整个过程不会触及 ~/.cache
    snapshot_download(
        repo_id=model_name,
        local_dir=local_path,
        local_dir_use_symlinks=False,  # 复制到 local_dir，避免指向中央缓存的链接
        cache_dir=hf_hub_cache         # 中央缓存也在 CACHE 下
    )

    print(f"All files saved to: {local_path}")
    print("Done (CPU-only, no model loaded).")

if __name__ == "__main__":
    main()