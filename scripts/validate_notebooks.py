"""Static checks; explicitly does not execute Spark or claim runtime validation."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = sorted((root / 'notebooks' / 'algorithms').glob('*.ipynb'))
assert len(paths) == 12, f'Expected 12 algorithm notebooks, found {len(paths)}'
code_count = 0
for path in paths:
    notebook = json.loads(path.read_text())
    assert notebook['nbformat'] == 4, path
    markdown = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'markdown')
    for number in range(1, 18):
        assert f'## {number}. ' in markdown, (path, number)
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        ast.parse(source, filename=f'{path.name}:cell{index}')
        assert index > 0 and notebook['cells'][index-1]['cell_type'] == 'markdown', path
        explanation = ''.join(notebook['cells'][index-1]['source'])
        for label in ['What and why', 'parameters', 'Expected output', 'Common errors']:
            assert label in explanation, (path, index, label)
        assert cell['outputs'] == [] and cell['execution_count'] is None, path
        assert 'toPandas(' not in source and '.rdd' not in source, path
        code_count += 1
print(f'PASS: {len(paths)} notebook files, 17 sections each, {code_count} explained Python cells; syntax/JSON/output checks only.')
