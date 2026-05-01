import pathlib
p = pathlib.Path('/opt/venv/lib/python3.12/site-packages/lerobot/async_inference/helpers.py')
src = p.read_text()
old = '    image_keys = list(filter(is_image_key, lerobot_obs))\n    # state\'s shape is expected as (B, state_dim)'
new = """    image_keys = list(filter(is_image_key, lerobot_obs))
    # --- PATCH: Remap obs image keys to policy image keys if mismatched ---
    _pik = sorted(policy_image_features.keys())
    if image_keys and _pik and not set(image_keys).intersection(set(_pik)):
        _osk = sorted(image_keys)
        _rmap = {pk: lerobot_obs[_osk[min(i, len(_osk)-1)]] for i, pk in enumerate(_pik)}
        for _ok in _osk:
            del lerobot_obs[_ok]
        lerobot_obs.update(_rmap)
        image_keys = sorted(filter(is_image_key, lerobot_obs))
    # --- END PATCH ---
    # state's shape is expected as (B, state_dim)"""
assert old in src, 'Patch target not found'
p.write_text(src.replace(old, new))
print('Patched helpers.py successfully')