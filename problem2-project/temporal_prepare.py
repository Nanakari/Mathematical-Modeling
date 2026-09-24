"""Download a pinned generic BERT-Mini checkpoint, never sentiment weights."""
import hashlib
import json
import urllib.request
from pathlib import Path
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent
MODEL = 'google/bert_uncased_L-4_H-256_A-4'
REVISION = '387825ce42dbb39b87911cdf8e383ee3b25184f8'


def main():
    folder = ROOT / 'models/bert_mini'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'provenance.json'
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding='utf-8'))
        assert record['revision'] == REVISION
        for name, info in record['files'].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == info['sha256']
        print('Pinned Mini already verified'); return
    downloaded = {}
    for name in ['config.json', 'vocab.txt', 'README.md', 'pytorch_model.bin']:
        target = folder / name
        url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
        print('Downloading', name, flush=True)
        urllib.request.urlretrieve(url, target)
        downloaded[name] = {'url': url, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    assert (folder / 'vocab.txt').read_bytes() == (ROOT / 'models/bert_tiny/vocab.txt').read_bytes()
    model = AutoModel.from_pretrained(str(folder), local_files_only=True)
    assert model.config.hidden_size == 256 and model.config.num_hidden_layers == 4
    model.save_pretrained(folder, safe_serialization=True)
    files = {}
    for name in ['config.json', 'vocab.txt', 'README.md', 'model.safetensors']:
        path = folder / name
        files[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps({'model': MODEL, 'revision': REVISION, 'downloaded': downloaded,
                                   'files': files, 'vocabulary_matches_verified_tiny': True}, indent=2), encoding='utf-8')
    print('Mini verified:', sum(p.numel() for p in model.parameters()), 'parameters', flush=True)


if __name__ == '__main__': main()
