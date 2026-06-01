"""
Agoya AI Trainer on Modal.com
Deploy: modal deploy agoyatrain.py
Run:    modal run agoyatrain.py
"""

import modal

app = modal.App("agoya-ai-trainer")

WALLET = "Wallet"
WORKER = "modal-h100"
GPU = "H100"
TIMEOUT = 86400

agoya_image = (
    modal.Image.from_registry(
        "skyhazee/agoya-trainer:latest",
        add_python="3.11",
    )
    .dockerfile_commands([
        "ENTRYPOINT []",
        "CMD []",
    ])
)


@app.function(
    gpu=GPU,
    image=agoya_image,
    timeout=TIMEOUT,
    scaledown_window=300,
)
def train():
    import subprocess
    import os

    os.environ["AKOYA_POOL_WALLET"] = WALLET
    os.environ["AKOYA_POOL_WORKER"] = WORKER
    os.environ["AKOYA_POOL_HOST"] = "pool-v2.akoyapool.com"
    os.environ["AKOYA_POOL_PORT"] = "443"
    os.environ["AKOYA_POOL_USE_TLS"] = "1"
    os.environ["AKOYA_GPU_INDICES"] = "all"
    os.environ["AKOYA_METRICS_PORT"] = "9100"
    os.environ["AKOYA_PEARL_GEMM_LIB"] = "/app/lib/libpearl_gemm_capi.so"
    os.environ["AKOYA_PEARL_MINING_LIB"] = "/app/lib/libpearl_mining_capi.so"

    cc_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )

    cc = cc_result.stdout.strip().split("\n")[0]
    major, minor = cc.split(".")

    print(f"[Modal] GPU compute: {major}.{minor}")

    lib_dir = "/app/lib"
    target = f"{lib_dir}/libpearl_gemm_capi.so"

    if int(major) == 12:
        src = "blackwell"
    elif int(major) == 9:
        src = "h100"
    elif int(major) == 8 and int(minor) == 9:
        src = "ada"
    else:
        src = "portable"

    lib_file = f"{lib_dir}/libpearl_gemm_capi_{src}.so"

    if os.path.lexists(target):
        os.unlink(target)

    os.symlink(lib_file, target)

    print(f"[Modal] Kernel: {src}")

    os.makedirs("/var/lib/akoya-miner", exist_ok=True)

    os.execv(
        "/app/akoya-miner",
        ["/app/akoya-miner", "mine-blocks"],
    )


@app.local_entrypoint()
def main():
    train.remote()
