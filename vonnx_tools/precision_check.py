"""Independent same-ONNX GPU reference; this is not an AP accuracy claim."""
from types import SimpleNamespace
import ast
import numpy as np
from .common import ROOT, manifest, engine, images, load_rgb, sha256

def stable_reference_nms(boxes,scores,iou_threshold):
    """Independent Torch greedy NMS with the public original-index tie contract."""
    import torch
    device=boxes.device
    boxes=boxes.cpu();scores=scores.cpu()
    order=torch.argsort(scores,descending=True,stable=True)
    keep=[]
    while order.numel():
        index=order[0];keep.append(index)
        rest=order[1:]
        if not rest.numel():break
        a=boxes[index];b=boxes[rest]
        inter=(torch.minimum(a[2:],b[:,2:])-torch.maximum(a[:2],b[:,:2])).clamp(min=0).prod(1)
        area_a=(a[2:]-a[:2]).clamp(min=0).prod()
        area_b=(b[:,2:]-b[:,:2]).clamp(min=0).prod(1)
        iou=inter/(area_a+area_b-inter).clamp(min=1e-12)
        order=rest[iou<=iou_threshold]
    return torch.stack(keep).to(device) if keep else torch.empty(0,dtype=torch.long,device=device)

def smoke(case):
    import torch
    import onnxruntime as ort
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from ultralytics.data.augment import LetterBox
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.models.yolo.pose.predict import PosePredictor
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from vonnx import PixelFormat
    from unittest.mock import patch
    import torchvision
    from ultralytics.utils.nms import TorchNMS
    from .validate import compare
    info=manifest(case['model']); shape=tuple(case['imgsz'])
    rgb=load_rgb(images(info['task']))
    if case.get('image_limit'):rgb=rgb[:case['image_limit']]
    summaries=[]; max_pre=0.; max_raw=0.; prepared=None; score_ties=[]; configurations=[]
    for batch in case.get('batch_limits',(1,2,4,8)):
        eng=engine(case['model'],shape,batch=batch,debug=True,precision=case.get('precision','fp16'),
                   model_path=case.get('model_path'))
        try:
            spec=eng.model_spec
            configurations.append(dict(max_batch_size=batch,fixed_batch_size=spec.fixed_batch_size,
                                       cuda_prefer_nhwc=spec.cuda_prefer_nhwc))
            if prepared is None:
                prepared=dict(path=spec.prepared_model_path,sha256=sha256(spec.prepared_model_path),
                              source_precision=spec.source_precision,effective_precision=spec.effective_precision,
                              input_dtype=spec.input_dtype,output_dtypes=spec.output_dtypes)
                session=ort.InferenceSession(spec.prepared_model_path, providers=[
                    ('CUDAExecutionProvider',dict(use_tf32=False,cudnn_conv_algo_search='HEURISTIC',
                                                  prefer_nhwc=int(spec.cuda_prefer_nhwc)))])
                metadata=session.get_modelmeta().custom_metadata_map
                cls={'detect':DetectionPredictor,'pose':PosePredictor,'segment':SegmentationPredictor}[info['task']]
                predictor=cls(overrides=dict(conf=.25,iou=.7,max_det=300,retina_masks=True,save=False,verbose=False))
                predictor.model=SimpleNamespace(names=ast.literal_eval(metadata['names']),end2end=False,
                    kpt_shape=ast.literal_eval(metadata.get('kpt_shape','[0, 0]')))
            for start in range(0,len(rgb),batch):
                inputs=rgb[start:start+batch]
                futures=[eng.submit_numpy(j,start+j,a,PixelFormat.RGB24) for j,a in enumerate(inputs)]
                actual=[f.result(timeout=30)[0] for f in futures]
                groups={}
                for j,r in enumerate(actual): groups.setdefault((r.batch_size,r.physical_batch_size),[]).append(j)
                tensors=[LetterBox(shape,auto=False)(image=np.ascontiguousarray(a[:,:,::-1]))[:,:,::-1].transpose(2,0,1).astype(np.float32)/255 for a in inputs]
                for (logical,physical),indices in groups.items():
                    assert len(indices)%logical==0
                    for offset in range(0,len(indices),logical):
                        chosen=indices[offset:offset+logical]
                        feed=np.stack([tensors[j] for j in chosen])
                        if physical>logical:feed=np.concatenate([feed,np.zeros((physical-logical,3,*shape),np.float32)])
                        if spec.input_dtype==10:feed=feed.astype(np.float16)
                        raw=session.run(None,{session.get_inputs()[0].name:feed})
                        if not all(np.isfinite(a).all() for a in raw):
                            raise AssertionError('Non-finite reference model output')
                        for pos,j in enumerate(chosen):
                            confidence=raw[0][pos,4:4+spec.classes].max(axis=0)
                            _,counts=np.unique(confidence[confidence>.25],return_counts=True)
                            score_ties.append(dict(image=start+j,batch=physical,
                                equal_score_groups=int((counts>1).sum())))
                        # Use official FP32 postprocessing on the same model output,
                        # matching vONNX's FP32 NMS/keypoint/mask accumulation.
                        prediction=[torch.from_numpy(a[:logical].astype(np.float32)).cuda() for a in raw]
                        predictor.batch=([str(start+j) for j in chosen],None,None)
                        with patch.object(torchvision.ops,'nms',stable_reference_nms),patch.object(TorchNMS,'nms',stable_reference_nms):
                            refs=predictor.postprocess(prediction if len(prediction)>1 else prediction[0],
                                torch.from_numpy(feed[:logical].astype(np.float32)).cuda(),
                                [np.ascontiguousarray(inputs[j][:,:,::-1]) for j in chosen])
                        for pos,j in enumerate(chosen):
                            result=actual[j]
                            assert (result.stream_id,result.frame_id)==(j,start+j)
                            own=np.asarray(result.debug_input,np.float32).reshape(3,*shape)
                            max_pre=max(max_pre,float(np.abs(own-tensors[j]).max()))
                            if max_pre>1e-4:raise AssertionError(f'Preprocess error {max_pre}')
                            for k,output in enumerate(raw):
                                reference=output[pos:pos+1].astype(np.float32)
                                value=np.asarray(result.debug_outputs[k],np.float32).reshape(reference.shape)
                                if not np.isfinite(value).all():raise AssertionError('Non-finite engine output')
                                np.testing.assert_allclose(value,reference,atol=1e-3,rtol=1e-2)
                                max_raw=max(max_raw,float(np.abs(value-reference).max()))
                            try:
                                summaries.append(compare(result,refs[pos],inputs[j].shape[:2]))
                            except AssertionError:
                                print('Reference diagnostic:',dict(batch=physical,index=start+j,
                                    max_raw=max_raw,own=[(b.x1,b.y1,b.x2,b.y2,b.score,b.class_id) for b in result.boxes],
                                    ref=refs[pos].boxes.data.cpu().tolist()),flush=True)
                                raise
            before=eng.resource_stats
            eng.pause();eng.resume()
            assert eng.resource_stats.initialization_count==before.initialization_count
            assert eng.resource_stats.explicit_device_allocations==before.explicit_device_allocations
        finally:eng.close()
    return dict(images=len(rgb),imgsz=shape,prepared_model=prepared,max_preprocess_error=max_pre,
                max_raw_error=max_raw,comparisons=summaries,ap_validated=False,
                equal_score_diagnostics=score_ties,
                batch_configurations=configurations,
                reference_nms='official postprocessing with independent stable original-index greedy NMS')
