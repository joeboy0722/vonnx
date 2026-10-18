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

def prepare(path, cache_dir=None, *, shape=None, fixed_batch=0, convert_fp16=True,
            specialization_order="before"):
    import onnx
    import onnxruntime as ort
    from onnxruntime.transformers.float16 import convert_float_to_float16, DEFAULT_OP_BLOCK_LIST
    path = Path(path).resolve()
    model = onnx.load(path, load_external_data=False)
    if specialization_order not in ("before", "after"):
        raise ValueError("specialization_order must be before or after")
    source_shape = []
    if shape is not None:
        if len(shape) != 2 or any(int(x) <= 0 for x in shape) or fixed_batch < 0:
            raise ValueError("Invalid specialization shape/batch")
        if len(model.graph.input) != 1:
            raise ValueError("Specialization requires one NCHW input")
        dims = model.graph.input[0].type.tensor_type.shape.dim
        source_shape = [d.dim_value if d.HasField('dim_value') else -1 for d in dims]
        if len(dims) != 4 or source_shape[1] != 3:
            raise ValueError("Specialization requires NCHW RGB input")
        for axis, size in ((2,int(shape[0])),(3,int(shape[1]))):
            if source_shape[axis] > 0 and source_shape[axis] != size:
                raise ValueError("imgsz does not match static ONNX shape")
        if source_shape[0] > 0 and fixed_batch and source_shape[0] != fixed_batch:
            raise ValueError("Specialization batch does not match source static batch")

    def specialize(graph_model):
        dims = graph_model.graph.input[0].type.tensor_type.shape.dim
        for axis, size in ((2,int(shape[0])),(3,int(shape[1]))):
            dims[axis].ClearField('dim_param'); dims[axis].dim_value=size
        if fixed_batch:
            dims[0].ClearField('dim_param'); dims[0].dim_value=fixed_batch
        # Exported intermediate shapes may still contain the original symbols.
        # Re-infer from the validated input instead of retaining stale hints.
        for g in graphs(graph_model.graph):
            del g.value_info[:]
        for i, output in enumerate(graph_model.graph.output):
            for j, dim in enumerate(output.type.tensor_type.shape.dim):
                dim.ClearField('dim_value');dim.dim_param=f'vonnx_out_{i}_{j}'
        return onnx.shape_inference.infer_shapes(graph_model, strict_mode=True, data_prop=True)
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
    settings = dict(schema=2, converter_sha256=digest(__file__), source_sha256=digest(path), external=external,
                    ort=ort.__version__, onnx=onnx.__version__, keep_io_types=True,
                    block_list=sorted(DEFAULT_OP_BLOCK_LIST), node_block_list=protected,
                    min_positive_val=1e-7, max_finite_val=1e4,
                    shape=list(shape) if shape is not None else None,fixed_batch=fixed_batch,
                    convert_fp16=convert_fp16,specialization_order=specialization_order)
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
        if shape is not None and specialization_order == 'before':
            model = specialize(model)
        converted = convert_float_to_float16(model, keep_io_types=True,node_block_list=protected) if convert_fp16 else model
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
        if shape is not None and specialization_order == 'after':
            converted = specialize(converted)
        onnx.checker.check_model(converted)
        # Metadata records transformation, not a promise about hardware accumulation.
        props = {p.key: p.value for p in converted.metadata_props}
        if convert_fp16:
            props['vonnx.fp16_source_sha256'] = settings['source_sha256']
            props['vonnx.fp16_conversion'] = json.dumps(settings, sort_keys=True)
        if shape is not None:
            props['vonnx.graph_specialized']='1'
            props['vonnx.source_input_shape']=','.join(map(str,source_shape))
            props['vonnx.source_sha256']=settings['source_sha256']
            props['vonnx.preparation']=json.dumps(settings,sort_keys=True)
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
    p.add_argument('--shape',type=int,nargs=2)
    p.add_argument('--fixed-batch',type=int,default=0)
    p.add_argument('--no-fp16',action='store_true')
    args = p.parse_args()
    try:
        result = prepare(args.source, args.cache_dir,shape=args.shape,fixed_batch=args.fixed_batch,
                         convert_fp16=not args.no_fp16)
        Path(args.response).write_text(str(result), encoding='utf-8')
    except Exception as e:
        Path(args.response).write_text('ERROR: ' + str(e), encoding='utf-8')
        raise
