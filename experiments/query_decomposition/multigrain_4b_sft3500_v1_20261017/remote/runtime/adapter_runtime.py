"""Focused ordinary-LoRA helpers. Imports do not load CUDA or model libraries."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

PROJECTIONS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
CONFIRMED_LORA = {'r': 32, 'lora_alpha': 64, 'lora_dropout': 0.05,
                  'target_modules': PROJECTIONS, 'bias': 'none', 'modules_to_save': None,
                  'use_rslora': False, 'use_dora': False, 'task_type': 'CAUSAL_LM'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def object_hash(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_identity(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    files = value.get('files')
    if not isinstance(files, list) or not files or value.get('manifest_sha256') != object_hash(files):
        raise ValueError('Invalid frozen base identity file manifest')
    if not isinstance(value.get('repo_id'), str) or not isinstance(value.get('revision'), str):
        raise ValueError('Base identity must record repository and frozen revision')
    if len(value['revision']) != 40 or any(c not in '0123456789abcdef' for c in value['revision']):
        raise ValueError('Base revision must be a frozen full commit SHA')
    if not isinstance(value.get('local_path'), str):
        raise ValueError('Base identity must record actual local model path')
    seen = set()
    for row in files:
        rel = Path(row.get('path', ''))
        if not row.get('path') or rel.is_absolute() or '..' in rel.parts or row['path'] in seen:
            raise ValueError('Invalid or duplicate base identity relative path')
        if type(row.get('bytes')) is not int or row['bytes'] < 0:
            raise ValueError('Invalid base identity byte count')
        digest = row.get('sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid base identity SHA256')
        seen.add(row['path'])
    return value


def verify_base_identity(identity_file, model_path, *, verify_bytes=True):
    identity = read_identity(identity_file)
    root = Path(model_path).resolve()
    if Path(identity['local_path']).resolve() != root:
        raise ValueError('Configured base path differs from frozen base identity')
    if (root / 'adapter_config.json').exists():
        raise ValueError('Frozen upstream directory must not implicitly contain an adapter')
    covered = {item['path'] for item in identity['files']}
    required = {'config.json', 'tokenizer_config.json'}
    required.update(str(p.relative_to(root)).replace('\\', '/') for p in root.rglob('*.safetensors'))
    required.update(str(p.relative_to(root)).replace('\\', '/') for p in root.glob('*.safetensors.index.json'))
    # Tokenizer JSON/vocabulary/model and native template are part of the input contract.
    required.update(p.name for p in root.iterdir() if p.is_file() and
                    (p.name.startswith('tokenizer') or p.name in ('vocab.json', 'merges.txt', 'special_tokens_map.json', 'chat_template.jinja')))
    if not required <= covered or not any(name.endswith('.safetensors') for name in covered):
        raise ValueError('Frozen identity does not cover required model and tokenizer files')
    for item in identity['files']:
        path = root / item['path']
        if any(part.is_symlink() for part in [path, *path.parents] if part == root or root in part.parents):
            raise ValueError('Base identity requires plain files and directories')
        if not path.is_file() or root not in path.resolve().parents or path.stat().st_size != item['bytes']:
            raise ValueError('Base identity missing/changed file: ' + item['path'])
        if verify_bytes and sha_file(path) != item['sha256']:
            raise ValueError('Base identity content changed: ' + item['path'])
    return identity


def lora_settings(value):
    settings = dict(CONFIRMED_LORA)
    settings.update(value or {})
    if set(settings) != set(CONFIRMED_LORA):
        raise ValueError('Only the explicit ordinary-LoRA configuration is supported')
    if set(settings['target_modules']) != set(PROJECTIONS) or len(settings['target_modules']) != 7:
        raise ValueError('LoRA must target exactly the seven confirmed layer projections')
    if settings['bias'] != 'none' or settings['modules_to_save'] is not None or settings['use_rslora'] or settings['use_dora']:
        raise ValueError('Only ordinary adapter-only LoRA is supported')
    if settings['task_type'] != 'CAUSAL_LM':
        raise ValueError('LoRA task must be CAUSAL_LM')
    settings['target_modules'] = sorted(settings['target_modules'])
    return settings


def validate_adapter_config(adapter_config, settings):
    actual = adapter_config.to_dict()
    for key, value in settings.items():
        observed = actual.get(key)
        if key == 'target_modules':
            observed = sorted(observed)
        if observed != value:
            raise ValueError(f'Saved adapter setting changed: {key}: {observed!r} != {value!r}')
    # These features change model structure or initialization beyond ordinary LoRA.
    for key in ('rank_pattern', 'alpha_pattern', 'layers_to_transform', 'layer_replication', 'trainable_token_indices'):
        if actual.get(key):
            raise ValueError('Unexpected extra adapter structure: ' + key)


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return get_peft_model_state_dict(model, adapter_name='default', save_embedding_layers=False)


def adapter_snapshot(model):
    return {key: tensor.detach().cpu().clone() for key, tensor in adapter_state(model).items()}


def compare_adapter_state(actual, expected, torch):
    if set(actual) != set(expected):
        raise ValueError(f'Adapter key set differs: missing={sorted(set(expected)-set(actual))}; extra={sorted(set(actual)-set(expected))}')
    for key, tensor in expected.items():
        current = actual[key]
        if current.shape != tensor.shape or current.dtype != tensor.dtype:
            raise ValueError('Adapter tensor shape/dtype differs: ' + key)
        if not torch.equal(current.detach().cpu(), tensor.detach().cpu()):
            raise ValueError('Adapter tensor values differ: ' + key)
    return {'adapter_keys': len(expected), 'exact_shapes_dtypes_values': True}


def attach_adapter(base_model, config, torch, train, adapter_path=None):
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    settings = lora_settings(config.get('lora'))
    peft_config = LoraConfig(**settings)
    if adapter_path:
        saved_config = LoraConfig.from_pretrained(str(adapter_path), local_files_only=True)
        validate_adapter_config(saved_config, settings)
        # Use the checked settings and explicit frozen base, never the adapter's implicit path.
    model = get_peft_model(base_model, peft_config, adapter_name='default', autocast_adapter_dtype=True)
    if adapter_path:
        weight_file = Path(adapter_path) / 'adapter_model.safetensors'
        if not weight_file.is_file() or weight_file.is_symlink():
            raise ValueError('A plain safetensors adapter weight file is required')
        saved = load_file(str(weight_file), device='cpu')
        expected = adapter_state(model)
        if set(saved) != set(expected):
            raise ValueError(f'Adapter keys missing/extra: missing={sorted(set(expected)-set(saved))}; extra={sorted(set(saved)-set(expected))}')
        for key, tensor in saved.items():
            if tensor.shape != expected[key].shape or tensor.dtype != torch.float32:
                raise ValueError('Adapter must preserve expected shape and FP32 dtype: ' + key)
        result = set_peft_model_state_dict(model, saved, adapter_name='default', ignore_mismatched_sizes=False)
        if result.unexpected_keys or any('lora_' in key for key in result.missing_keys):
            raise ValueError(f'Adapter strict load failed: {result}')
        compare_adapter_state(adapter_state(model), saved, torch)
    for name, parameter in model.named_parameters():
        is_adapter = '.lora_A.default.' in name or '.lora_B.default.' in name
        parameter.requires_grad_(bool(train and is_adapter))
        if is_adapter and parameter.dtype != torch.float32:
            raise RuntimeError('PEFT adapter is not FP32: ' + name)
        if not is_adapter and parameter.requires_grad:
            raise RuntimeError('Frozen base became trainable: ' + name)
    return model, {'settings': settings, 'loaded_adapter': str(Path(adapter_path).resolve()) if adapter_path else None,
                   'strict_adapter_shapes_values_verified': bool(adapter_path),
                   'adapter_dtype': 'torch.float32', 'base_frozen': True}


def trainable_parameter_spec(model):
    return [{'name': name, 'shape': list(parameter.shape), 'dtype': str(parameter.dtype)}
            for name, parameter in model.named_parameters() if parameter.requires_grad]


def validate_trainable_parameters(model, torch):
    spec = trainable_parameter_spec(model)
    if not spec or any(not ('.lora_A.default.' in row['name'] or '.lora_B.default.' in row['name']) for row in spec):
        raise RuntimeError('Optimizer may contain only the confirmed LoRA adapter tensors')
    if any(row['dtype'] != str(torch.float32) for row in spec):
        raise RuntimeError('Trainable adapter parameters must remain FP32')
    return spec


def optimizer_report(optimizer, model, torch):
    return {'optimizer': 'torch.optim.AdamW on FP32 trainable LoRA adapters only',
            'adapter_parameter_dtypes': sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
            'optimizer_tensors': sorted({str(v.dtype) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)}),
            'separate_master_weights': False, 'scaler': None}


def frozen_base_fingerprint(model, torch):
    """Hash actual in-memory base tensor bytes in bounded CPU chunks, excluding LoRA."""
    rows = []
    for name, parameter in model.named_parameters():
        if '.lora_A.' in name or '.lora_B.' in name:
            continue
        if parameter.requires_grad or parameter.grad is not None:
            raise RuntimeError('Frozen base has training/gradient state: ' + name)
        digest = hashlib.sha256()
        flat = parameter.detach().reshape(-1)
        for begin in range(0, flat.numel(), 1024 * 1024):
            raw = flat[begin:begin + 1024 * 1024].contiguous().cpu().view(torch.uint8)
            digest.update(raw.numpy().tobytes())
        rows.append({'name': name, 'shape': list(parameter.shape), 'dtype': str(parameter.dtype), 'sha256': digest.hexdigest()})
    return {'parameters': rows, 'fingerprint_sha256': object_hash(rows)}
