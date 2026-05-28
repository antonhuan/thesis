from huggingface_hub import HfApi
api = HfApi()

datasets = ["ant0nh/pnp_200"]

for ds in datasets:
    refs = api.list_repo_refs(ds, repo_type="dataset")
    tags = [t.name for t in refs.tags]
    print(f"{ds}: {tags}")
    if not tags:
        api.create_tag(ds, repo_type="dataset", tag="v3.0")  # match your codebase_version
        print(f"  -> created v3.0 tag")