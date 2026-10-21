"""Audit and import remote dev200 decomposition as data; no remote code runs."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import (
    ROOT, OUTPUT, canonical, digest, utc, write_json,
)
from .common import HERE as PLAN, RUN, read as read_json, rows


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def mechanical(raw):
    """Check JSON fields and prior-reference legality, without semantic scoring."""
    try:
        target = json.loads(raw)
    except (TypeError, ValueError):
        return None, ['json_parse_failure']
    if not isinstance(target, dict) or set(target) != {'steps', 'goal'}:
        return target, ['invalid_qdmr_fields']
    steps, goal = target['steps'], target['goal']
    if not isinstance(steps, list) or not steps or type(goal) is not int or not 1 <= goal <= len(steps):
        return target, ['invalid_steps_or_goal']
    errors = []
    for number, step in enumerate(steps, 1):
        if (not isinstance(step, dict) or set(step) != {'id', 'request'}
                or type(step['id']) is not int or step['id'] != number
                or not isinstance(step['request'], str) or not step['request'].strip()):
            errors.append(f'invalid_step:{number}')
            continue
        references = re.findall(r'#(\d+)', step['request'])
        if step['request'].count('#') != len(references):
            errors.append(f'malformed_reference:{number}')
        if any(not 1 <= int(ref) < number for ref in references):
            errors.append(f'invalid_prior_reference:{number}')
    return target, errors


def import_return(*, archive, preparation, question_file, destination_root, event):
    """Validate a frozen 200-question package without changing running methods."""
    package = preparation / 'remote_package'
    approved = json.loads((package / 'PACKAGE_MANIFEST.json').read_text(encoding='utf8'))
    expected_rows = [json.loads(line) for line in question_file.read_text(encoding='utf8').splitlines() if line.strip()]
    expected = {row['qid']: row['question'] for row in expected_rows}
    if len(expected_rows) != len(expected) or len(expected) != approved["queries"]:
        raise ValueError('Frozen question denominator differs')
    if (sha(question_file.read_bytes()) != approved['local_source_sha256']
            or question_file.resolve() != (ROOT / approved['local_source']).resolve()):
        raise ValueError('Frozen questions changed after packaging')
    run = 'runs/' + approved['run_name']
    details, counts, step_counts, issues = [], Counter(), Counter(), Counter()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(set(names)) != len(names) or sum(info.file_size for info in z.infolist()) > 100_000_000:
            raise ValueError('Duplicate entries or unexpected return size')
        for info in z.infolist():
            path = PurePosixPath(info.filename.replace('\\', '/'))
            if (path.is_absolute() or '..' in path.parts or any(':' in part for part in path.parts)
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError('Unsafe return path')
        if z.testzip() is not None:
            raise ValueError('Return archive CRC failed')
        manifest = json.loads(z.read('RETURN_FILE_MANIFEST.json'))
        actual_files = {info.filename for info in z.infolist() if not info.is_dir() and info.filename != 'RETURN_FILE_MANIFEST.json'}
        if len({row['path'] for row in manifest}) != len(manifest) or {row['path'] for row in manifest} != actual_files:
            raise ValueError('Return manifest does not cover exactly all files')
        for row in manifest:
            raw = z.read(row['path'])
            if len(raw) != row['bytes'] or sha(raw) != row['sha256']:
                raise ValueError('Returned file differs from integrity manifest')
        if json.loads(z.read('execution/package/PACKAGE_MANIFEST.json')) != approved:
            raise ValueError('Approved package manifest changed')
        for name, expected_sha in approved['files'].items():
            if sha(z.read('execution/package/' + name)) != expected_sha:
                raise ValueError('Submitted package file changed: ' + name)
        submitted_archives = list(preparation.glob('qdmr4b_multidata_formal1500_remote_package.zip'))
        if len(submitted_archives) != 1:
            raise ValueError('Expected one original submitted package archive')
        submitted_archive = submitted_archives[0]
        if z.read('execution/' + submitted_archive.name) != submitted_archive.read_bytes():
            raise ValueError('Submitted package archive changed')
        input_bytes = (package / 'queries200.query_only.jsonl').read_bytes()
        if z.read(run + '/queries200.query_only.jsonl') != input_bytes:
            raise ValueError('Actual predictor input changed')
        cfg = json.loads(z.read(run + '/predict200.json'))
        proposed = json.loads((package / 'predict200.proposed.json').read_text(encoding='utf8'))
        path_fields = {'adapter_path', 'model_path', 'base_identity_file', 'input_file', 'output_dir'}
        if {k: v for k, v in cfg.items() if k not in path_fields} != {k: v for k, v in proposed.items() if k not in path_fields}:
            raise ValueError('Instruction or inference configuration changed')
        adapter = json.loads(z.read('checkpoint_identity/checkpoint-000148/manifest.json'))
        base = json.loads(z.read('model_identity/BASE_MODEL_IDENTITY.json'))
        if adapter['manifest_sha256'] != approved['adapter_manifest_sha256'] or base['manifest_sha256'] != approved['base_manifest_sha256']:
            raise ValueError('Base or checkpoint identity changed')
        combined = digest({'base_model_manifest': base['manifest_sha256'], 'adapter_manifest': adapter['manifest_sha256']})
        if combined != approved['expected_model_manifest']:
            raise ValueError('Combined model identity differs')
        before = json.loads(z.read('execution/EXECUTION_IDENTITY_BEFORE.json'))
        prior_sources = json.loads((ROOT / 'outputs/query_decomposition/ircot_alignment_v1/returned/ircot_alignment_v1_500/execution/PREDICTOR_SOURCE_IDENTITY.json').read_text(encoding='utf8'))
        source_hashes = {PurePosixPath(row['path']).name: row['sha256'] for row in prior_sources}
        if {PurePosixPath(row['path']).name: row['sha256'] for row in before['source_files']} != source_hashes:
            raise ValueError('Predictor sources differ from the previously approved deployment')
        for row in before['source_files']:
            name = PurePosixPath(row['path']).name
            raw = z.read('execution/predictor_source/' + name)
            if len(raw) != row['bytes'] or sha(raw) != row['sha256']:
                raise ValueError('Execution source differs from recorded preflight')
        template = json.loads(z.read(run + '/predictions/prompt_template.json'))
        if (template['template_policy'] != cfg['template_policy'] or template['instruction'] != cfg['instruction']
                or template['enable_thinking'] is not None
                or 'enable_thinking' in template['chat_template'] or '<think>' in template['chat_template']):
            raise ValueError('Native nonthinking template changed')
        prediction_name = run + '/predictions/predictions.jsonl'
        predictions = [json.loads(line) for line in z.read(prediction_name).decode('utf8').splitlines() if line.strip()]
        if len(predictions) != len(expected) or len({r['id'] for r in predictions}) != len(expected) or {r['id'] for r in predictions} != set(expected):
            raise ValueError('Returned question IDs are incomplete or duplicated')
        for row in predictions:
            if row['query'] != expected[row['id']] or row['checkpoint_sha256_or_manifest'] != combined:
                raise ValueError('Returned question text or model identity differs')
            decode = row['decoding_config']
            if (decode.get('do_sample') is not False or decode.get('num_beams') != 1
                    or decode.get('max_new_tokens') != 1024 or row['template_policy'] != cfg['template_policy']
                    or row.get('enable_thinking') is not None or row['scope'] != cfg['scope']):
                raise ValueError('Returned decoding/template policy differs')
            target, errors = mechanical(row['raw_output'])
            if target != row['parsed_target']:
                raise ValueError('Parsed target differs from preserved raw output')
            counts['generation_failures'] += bool(row.get('generation_error'))
            counts['parse_failures'] += bool(row.get('parse_error'))
            counts['truncated_outputs'] += bool(row.get('truncated'))
            counts['records_with_mechanical_issues'] += bool(errors)
            issues.update(errors)
            if isinstance(target, dict) and isinstance(target.get('steps'), list):
                step_counts[len(target['steps'])] += 1
            details.append({'qid': row['id'], 'issues': errors})
        summary = json.loads(z.read(run + '/predictions/SUMMARY.json'))
        identity = json.loads(z.read(run + '/predictions/IDENTITY.json'))
        if (summary['status'] != 'completed' or summary['total_inputs'] != len(expected) or summary['completed_attempts'] != len(expected)
                or summary['input_hash'] != sha(input_bytes) or identity['input_sha256'] != sha(input_bytes)
                or identity['instruction'] != cfg['instruction'] or identity['model_manifest'] != combined
                or set(identity['query_order']) != set(expected)):
            raise ValueError('Completion, prompt or input identity differs')
        for name in ['generation_failures', 'parse_failures', 'truncated_outputs']:
            if summary[name] != counts[name]:
                raise ValueError('Summary differs from raw records')
        # All checks completed before importing any archive file.
        for info in z.infolist():
            destination = (destination_root / info.filename).resolve()
            if not destination.is_relative_to(destination_root.resolve()):
                raise ValueError('Resolved return path escapes destination')
            if info.is_dir():
                continue
            raw = z.read(info)
            if destination.exists() and destination.read_bytes() != raw:
                raise ValueError('Previously imported file differs')
        for info in z.infolist():
            destination = destination_root / info.filename
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(z.read(info))
    result = {'utc': utc(), 'event': event, 'passed': True,
              'archive': str(archive), 'archive_sha256': sha(archive.read_bytes()),
              'records': len(expected), 'unique_ids': len(expected), 'files_verified': len(manifest),
              'input_id_query_identity_verified': True, 'settings_unchanged': True,
              'combined_model_manifest': combined, 'raw_record_counts': dict(counts),
              'predictor_source_matches_previous_approved_deployment': True,
              'thinking_field': 'null; native nonthinking template has no enable_thinking parameter',
              'mechanical_issue_counts': dict(issues), 'step_count_distribution': dict(step_counts),
              'predictions': str(destination_root / prediction_name),
              'semantic_decomposition_accuracy_evaluated': False,
              'downstream_retrieval_or_qa_executed': False, 'remote_scripts_executed_locally': False,
              'baseline_sources_inputs_or_running_campaign_changed': False}
    write_json(preparation / 'REMOTE_RETURN_AUDIT.json', result)
    write_json(preparation / 'LOCAL_MECHANICAL_DIAGNOSTICS.json', details)
    with (PLAN / 'changes.jsonl').open('a', encoding='utf8') as stream:
        stream.write(canonical(result) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    return result



def main():
    preparation=RUN/'remote1500'
    archive=ROOT.parent/'remote/comprehensive_formal1500_20261015_returned.zip'
    if not archive.exists():
        return {'status':'waiting_for_manual_remote_return','expected_archive':str(archive)}
    audit=import_return(archive=archive,preparation=preparation,question_file=preparation/'questions1500.query_only.jsonl',
        destination_root=RUN/'returned',event='multidata1500_remote_return_verified')
    mapping=read_json(preparation/'LOCAL_ID_MAPPING.json')
    exported={dataset:[] for dataset in ['hotpotqa','2wiki','musique']}
    for r in rows(audit['predictions']):
        local=mapping[r['id']]
        exported[local['dataset']].append({**r,'remote_id':r['id'],'id':local['qid']})
    for dataset,items in exported.items():
        if len(items)!=500:raise ValueError('Wrong dataset denominator')
        path=preparation/(dataset+'.decompositions.jsonl')
        data=''.join(canonical(r)+'\n' for r in items)
        if path.exists() and path.read_text(encoding='utf8')!=data:raise ValueError('Accepted decompositions differ')
        if not path.exists():path.write_text(data,encoding='utf8')
    return audit

if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False))

