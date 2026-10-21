"""Keep frozen methods; accept the exact approved multidataset archive name."""
from .common import HERE, RUN, read, sha, log
from . import accept_return, accept_return_multidata, run_execution_v3


def install_import_compatibility():
    epoch = read(RUN / 'REMOTE_IMPORT_COMPATIBILITY_EPOCH.json')
    if (sha(HERE / 'accept_return.py') != epoch['original_sha256']
            or sha(HERE / 'accept_return_multidata.py') != epoch['compatibility_sha256']):
        raise ValueError('Remote import compatibility source changed')
    original = (HERE / 'accept_return.py').read_text(encoding='utf8')
    compatible = (HERE / 'accept_return_multidata.py').read_text(encoding='utf8')
    old = "preparation.glob('qdmr4b_hotpot_*_remote_package.zip')"
    new = "preparation.glob('qdmr4b_multidata_formal1500_remote_package.zip')"
    if original.count(old) != 1 or original.replace(old, new) != compatible:
        raise ValueError('Compatibility patch changes more than archive filename')
    package = read(RUN / 'remote1500/PACKAGE_AUDIT.json')
    if sha(package['archive']) != epoch['submitted_archive_sha256']:
        raise ValueError('Original submitted archive changed')
    accept_return.import_return = accept_return_multidata.import_return
    log('multidata_remote_import_compatibility_activated', epoch=epoch['id'])


def main():
    install_import_compatibility()
    run_execution_v3.main()


if __name__ == '__main__':
    main()
