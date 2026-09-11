"""Read-only dataset audit. Run with conda run -n lightlytrain python reports/face_wider_combined_audit/audit.py."""
import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[2]
B=R/'datasets/face_detect/face_yolo_wider'; C=B.parent/'face_yolo_wider_combined_v1'
out={}; cache={}
def boxes(p):
    a=np.array([list(map(float,l.split())) for l in p.read_text().splitlines() if l.strip()])
    return a.reshape(-1,5)
def stats(paths):
    arrays=[boxes(p) for p in paths]; a=np.concatenate(arrays); s=a[:,3:5].min(1)*640
    return dict(images=len(paths),boxes=len(a),short_quantiles=np.percentile(s,[10,50,90]).tolist(),short_bins=np.histogram(s,[0,8,16,32,float('inf')])[0].tolist(),images_over100=sum(len(x)>100 for x in arrays),images_over300=sum(len(x)>300 for x in arrays),excess100=sum(max(0,len(x)-100) for x in arrays),invalid=int(np.sum(~np.isfinite(a).all(1)|(a[:,3]<=0)|(a[:,4]<=0))))
for name,root in [('original',B),('combined',C)]:
    for split in ['train','val','test']:
        out[name+'_'+split]=stats(sorted((root/'labels'/split).glob('*.txt')))
for prefix in ['tile_','cp_']:
    out[prefix]=stats(sorted((C/'labels/train').glob(prefix+'*.txt')))
for split in ['val','test']:
    checks={}
    for kind in ['labels','images']:
        src={p.name:p for p in (B/kind/split).iterdir() if p.is_file()}; dst={p.name:p for p in (C/kind/split).iterdir() if p.is_file()}
        checks[kind]=dict(same_names=src.keys()==dst.keys(),different_bytes=sum(hashlib.sha256(p.read_bytes()).digest()!=hashlib.sha256(dst[n].read_bytes()).digest() for n,p in src.items() if n in dst))
    out[split+'_identity']=checks
manifest=[json.loads(l) for l in (C/'augmentation_manifest.jsonl').read_text().splitlines()]
train_names={p.name for p in (B/'images/train').iterdir()}; failures=[]; drops=[]; keptpartial=0; affected=0; contributors=Counter(); examples=[]
for row in manifest:
    names=[row['source_image']] if row['kind']=='tile' else [row['recipient_image']]+[p['donor_image'] for p in row['pastes']]
    failures.extend(n for n in names if n not in train_names)
    if row['kind']!='tile': continue
    n=row['source_image']; contributors[n]+=1
    if n not in cache:
        with Image.open(B/'images/train'/n) as im: w,h=im.size
        a=boxes(B/'labels/train'/Path(n).with_suffix('.txt')); xy=a[:,1:3]*[w,h]; wh=a[:,3:5]*[w,h]; cache[n]=((a[:,1:3]-a[:,3:5]/2)*[w,h],(a[:,1:3]+a[:,3:5]/2)*[w,h])
    lo,hi=cache[n]; crop=np.array(row['crop_xyxy']); il=np.maximum(lo,crop[:2]); ih=np.minimum(hi,crop[2:]); wh=np.maximum(0,ih-il); area=wh.prod(1); ratio=area/((hi-lo).prod(1)); side=crop[2]-crop[0]
    mask=(ratio>0)&(ratio<0.5); keptpartial+=int(((ratio>=0.5)&(ratio<0.999)).sum()); affected+=int(mask.any())
    for i in np.where(mask)[0]:
        short=float(wh[i].min()*640/side); drops.append((float(ratio[i]),short))
        if ratio[i]>=0.25 and short>=8: examples.append(dict(tile=row['output_image'],source=n,crop=row['crop_xyxy'],dropped_box=lo[i].tolist()+hi[i].tolist(),visible_fraction=float(ratio[i]),visible_short640=short))
out['tile_boundary']=dict(affected_tiles=affected,dropped_visible_boxes=len(drops),dropped_fraction25_short8=sum(r>=.25 and s>=8 for r,s in drops),kept_partial_boxes=keptpartial,source_images=len(contributors),source_reuse=dict(Counter(contributors.values())),examples=sorted(examples,key=lambda x:-x['visible_short640'])[:8])
out['manifest_nontrain_sources']=len(failures)
out['cp_pasted_boxes']=sum(len(r['pastes']) for r in manifest if r['kind']=='copy_paste')
# Check direct filename overlap of base splits.
sets={s:{p.name for p in (B/'images'/s).iterdir()} for s in ['train','val','test']}
out['split_name_overlap']={a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
path=Path(__file__).with_name('audit_results.json'); path.write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(json.dumps(out,ensure_ascii=False,indent=2))
