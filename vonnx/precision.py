"""Initialization-only ONNX FP16 conversion, shared by Python and native SDK.

Executable as a standalone script: no native vONNX import is needed.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import time

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def graphs(graph):
    import onnx
    yield graph
    for node in graph.node:
        for a in node.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                yield from graphs(a.g)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for g in a.graphs:
                    yield from graphs(g)

def prepare(path, cache_dir=None):
    import onnx
    import onnxruntime as ort
    from onnxruntime.transformers.float16 import convert_float_to_float16, DEFAULT_OP_BLOCK_LIST
    path = Path(path).resolve()
    model = onnx.load(path, load_external_data=False)
    # YOLOv8 coordinate/DFL decoding is sensitive to half rounding. Keep the
    # decoder in FP32 while convolution branches and the backbone use FP16.
    head_prefixes = {n.name.split('/cv')[0]+'/' for n in model.graph.node
                     if re.search(r'/cv[234]\.\d+/', n.name)}
    protected = [n.name for n in model.graph.node if n.name and
                 any(n.name.startswith(p) for p in head_prefixes) and
                 not re.search(r'/cv[234]\.\d+/', n.name) and '/proto/' not in n.name]
    external = {}
    for g in graphs(model.graph):
        tensors = list(g.initializer)
        for n in g.node:
            for a in n.attribute:
                if a.HasField('t'): tensors.append(a.t)
                tensors.extend(a.tensors)
        for tensor in tensors:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                location = dict((x.key, x.value) for x in tensor.external_data).get('location')
                if not location: raise ValueError('External tensor is missing its location')
                external[location] = digest(path.parent / location)
    settings = dict(schema=1, converter_sha256=digest(__file__), source_sha256=digest(path), external=external,
                    ort=ort.__version__, onnx=onnx.__version__, keep_io_types=True,
                    block_list=sorted(DEFAULT_OP_BLOCK_LIST), node_block_list=protected,
                    min_positive_val=1e-7, max_finite_val=1e4)
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    base = Path(cache_dir) if cache_dir else Path(os.environ.get('LOCALAPPDATA', Path.home()/'.cache'))/'vonnx/models'
    base.mkdir(parents=True, exist_ok=True)
    target = base / (key + '.onnx')
    record = base / (key + '.json')
    lock = base / (key + '.lock')
    start = time.monotonic()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() - start > 120:
                raise TimeoutError(f'Conversion cache lock timed out; inspect stale lock: {lock}')
            time.sleep(.05)
    try:
        if target.exists() and record.exists():
            info = json.loads(record.read_text(encoding='utf-8'))
            if info.get('settings') == settings and info.get('sha256') == digest(target):
                return target
        model = onnx.load(path, load_external_data=True)
        # Inline external weights so the atomic cache artifact is self contained.
        onnx.external_data_helper.convert_model_from_external_data(model)
        converted = convert_float_to_float16(model, keep_io_types=True,node_block_list=protected)
        # The ORT converter can emit the same boundary Cast more than once when
        # several blocked nodes consume one tensor. Merge identical Casts only.
        for g in graphs(converted.graph):
            seen={}; kept=[]
            for node in g.node:
                duplicate=False
                for output in node.output:
                    if output and output in seen:
                        prior=seen[output]
                        if (node.op_type==prior.op_type=='Cast' and list(node.input)==list(prior.input)
                            and list(node.output)==list(prior.output) and node.attribute==prior.attribute):
                            duplicate=True
                        else: raise ValueError('Conflicting converted tensor producer: '+output)
                if not duplicate:
                    kept.append(node)
                    for output in node.output:
                        if output:seen[output]=node
            del g.node[:];g.node.extend(kept)
        from onnxruntime.transformers.onnx_model import OnnxModel
        OnnxModel(converted).topological_sort()
        onnx.checker.check_model(converted)
        # Metadata records transformation, not a promise about hardware accumulation.
        props = {p.key: p.value for p in converted.metadata_props}
        props['vonnx.fp16_source_sha256'] = settings['source_sha256']
        props['vonnx.fp16_conversion'] = json.dumps(settings, sort_keys=True)
        onnx.helper.set_model_props(converted, props)
        fd, name = tempfile.mkstemp(prefix=key, suffix='.onnx.tmp', dir=base)
        os.close(fd)
        temporary = Path(name)
        try:
            onnx.save(converted, temporary)
            info = dict(settings=settings, sha256=digest(temporary))
            record_tmp = temporary.with_suffix('.json.tmp')
            record_tmp.write_text(json.dumps(info, sort_keys=True), encoding='utf-8')
            os.replace(temporary, target)
            os.replace(record_tmp, record)
        finally:
            temporary.unlink(missing_ok=True)
        return target
    finally:
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source')
    p.add_argument('response')
    p.add_argument('--cache-dir')
    args = p.parse_args()
    try:
        result = prepare(args.source, args.cache_dir)
        Path(args.response).write_text(str(result), encoding='utf-8')
    except Exception as e:
        Path(args.response).write_text('ERROR: ' + str(e), encoding='utf-8')
        raise
