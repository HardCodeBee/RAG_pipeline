"""Read-only resource preflight; never imports an encoder or initializes a GPU."""
from pathlib import Path

from . import runtime_config as cfg


def inspect_resources():
    checks = []
    for dataset in cfg.DATASETS:
        r = cfg.resource(dataset)
        bge, corpus = r['dense']['bge'], r['corpus']
        manifests = [(Path(corpus['manifest']), corpus['manifest_sha256']),
                     (Path(bge['build']) / ('manifest.json' if dataset == 'hotpotqa' else 'DENSE_FROZEN.json'),
                      bge['manifest_sha256'])]
        if bge.get('embeddings_manifest_sha256'):
            manifests.append((Path(bge['build']) / bge.get('manifest_name', 'embeddings/manifest.json'),
                              bge['embeddings_manifest_sha256']))
        for path, expected in manifests:
            checks.append({'dataset': dataset, 'kind': 'manifest_sha256',
                           'path': str(path), 'passed': path.is_file() and cfg.common.sha(path) == expected})
        for name, path in [('chunks', corpus['chunks']), ('offsets', corpus['offsets']),
                           ('query_model_snapshot', bge['query_snapshot'])]:
            checks.append({'dataset': dataset, 'kind': name, 'passed': Path(path).exists()})
        checks.append({'dataset': dataset, 'kind': 'BGE_model_identity',
                       'passed': bge['query_model'] == 'BAAI/bge-small-en-v1.5'
                                 and bge['query_revision'] == '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
                                 and bge['space']['dimension'] == 384})
    spec = cfg.read(cfg.common.HERE / 'methods/hybrid_rerank/RERANKER_FROZEN.json')
    checks.append({'kind': 'frozen_reranker_assets_exist',
                   'passed': all((Path(spec['snapshot']) / name).is_file() for name in spec['file_sha256'])})
    return {'passed': all(c['passed'] for c in checks), 'checks': checks,
            'scope': 'frozen resource manifests, identity, and asset presence; no binary index/model loading',
            'full_dense_parts_and_model_integrity': 'unchanged frozen loaders verify at actual approved execution',
            'actual_target_hardware_verified': False, 'model_loads': 0, 'retrieval_requests': 0}
