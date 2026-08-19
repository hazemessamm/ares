from huggingface_hub import snapshot_download


def load_ares_ec_moe_85k():
    repo_id = "HazemLab/ares-ec-moe-85k"
    path = snapshot_download(repo_id, repo_type="model")
    model = None
