import json
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[2]; B=R/'datasets/face_detect/face_yolo_wider'; C=B.parent/'face_yolo_wider_combined_v1'
def read(p): return np.array([list(map(float,l.split())) for l in p.read_text().splitlines() if l.strip()]).reshape(-1,5)
cache={}; mismatch=0; sizes=[]; source_counts=[]
for line in (C/'augmentation_manifest.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r['kind']!='tile':continue
 n=r['source_image']
 if n not in cache:
  with Image.open(B/'images/train'/n) as im:w,h=im.size
  a=read(B/'labels/train'/Path(n).with_suffix('.txt')); xy=a[:,1:3]*[w,h]; wh=a[:,3:]*[w,h]; cache[n]=((a[:,1:3]-a[:,3:5]/2)*[w,h],(a[:,1:3]+a[:,3:5]/2)*[w,h])
 lo,hi=cache[n]; crop=np.array(r['crop_xyxy']); side=crop[2]-crop[0]; sizes.append(side); source_counts.append(len(lo))
 il=np.maximum(lo,crop[:2]); ih=np.minimum(hi,crop[2:]); wh=np.maximum(0,ih-il); ratio=wh.prod(1)/(hi-lo).prod(1); keep=ratio>=.5
 expected=np.column_stack([np.zeros(keep.sum()),((il[keep]+ih[keep])/2-crop[:2])/side,wh[keep]/side]); actual=read(C/'labels/train'/Path(r['output_image']).with_suffix('.txt'))
 if expected.shape!=actual.shape or not np.allclose(expected,actual,atol=1e-6,rtol=0): print("MISMATCH",r["output_image"],expected.shape,actual.shape, "near_half",ratio[np.abs(ratio-.5)<1e-5].tolist())
 mismatch+=int(expected.shape!=actual.shape or not np.allclose(expected,actual,atol=1e-6,rtol=0))
print(json.dumps(dict(reconstruction_mismatch=mismatch,tiles= len(sizes),tile_side_quantiles=np.percentile(sizes,[0,10,50,90,100]).tolist(),tiles_from_sources_over100=sum(n>100 for n in source_counts),tiles_from_sources_over300=sum(n>300 for n in source_counts)),indent=2))
assert mismatch==0
