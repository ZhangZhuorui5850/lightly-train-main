import json,re
from pathlib import Path
D=Path(__file__).resolve().parent
md=(D/'report_source.md').read_text(); sections=re.split(r'(?=^## )',md,flags=re.M)
sources=[{'id':'audit','label':'本地数据全量审计','path':'reports/face_wider_combined_audit/audit_results.json','query':{'engine':'Python / NumPy / Pillow','description':'全量标签统计，按manifest重建裁剪交集，SHA-256比较val/test。','tables_used':['datasets/face_detect/face_yolo_wider','datasets/face_detect/face_yolo_wider_combined_v1'],'metric_definitions':['640等效短边=640×min(归一化宽,归一化高)','边界候选=原框和tile有交集且保留比例小于0.5']}},{'id':'screenshots','label':'用户提供的服务器结果截图','query':{'description':'本会话2026-09-10提供的metrics_summary.json和run_meta.json截图手工转录；两组vits16由用户确认。','tables_used':['原始模型SAHI metrics_summary','combined全图 metrics_summary','combined SAHI metrics_summary']}}]
blocks=[]; tables=[]; datasets={}
for i,body in enumerate(sections):
 # Convert markdown tables into canonical native tables, preserving prose order.
 parts=re.split(r'(^\|.*(?:\n\|.*)+)',body,flags=re.M)
 for j,p in enumerate(parts):
  if not p.strip():continue
  if p.startswith('|'):
   lines=p.strip().splitlines(); heads=[v.strip() for v in lines[0].strip('|').split('|')]; rows=[]
   for line in lines[2:]:rows.append({f'c{k}':v.strip() for k,v in enumerate(line.strip('|').split('|'))})
   tid=f't{i}_{j}'; datasets[tid]=rows
   titles={2:'三组实测检测指标',3:'训练数据构成',5:'640等效短边分布'}
   title='检测指标对照' if '指标' in heads[0] else '训练数据构成' if '子集' in heads[0] else '640等效短边分布'
   tables.append(dict(id=tid,title=title,dataset=tid,sourceId='screenshots' if '指标' in heads[0] else 'audit',columns=[dict(field=f'c{k}',label=h) for k,h in enumerate(heads)],density='spacious',layout='full'))
   blocks.append(dict(id=tid,type='table',tableId=tid))
  else:blocks.append(dict(id=f'b{i}_{j}',type='markdown',body=p.strip()))
manifest=dict(version=1,surface='report',title=md.splitlines()[0][2:],generatedAt='2026-09-10T00:00:00Z',blocks=blocks,tables=tables,charts=[],cards=[],sources=sources)
payload=dict(surface='report',manifest=manifest,snapshot=dict(version=1,generatedAt='2026-09-10T00:00:00Z',status='ready',datasets=datasets),sources=sources)
(D/'artifact.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
audit=json.loads((D/'audit_results.json').read_text())
rows=[]
for k,label in enumerate(['<8 px','8–16 px','16–32 px','≥32 px']):
 row={'size':label}
 for key in ['original_train','combined_train','original_test']:row[key]=audit[key]['short_bins'][k]/audit[key]['boxes']
 rows.append(row)
payload['snapshot']['datasets']['scales']=rows
payload['manifest']['charts']=[dict(id='scales',title='combined训练框向较大尺度迁移',subtitle='在线增强前；各子集框数占比；640等效短边',type='bar',dataset='scales',sourceId='audit',xField='size',series=[dict(field=k,label=v) for k,v in [('original_train','原始train'),('combined_train','combined train'),('original_test','test')]],valueFormat='percent',layout='full',xAxisTitle='640等效短边',yAxisTitle='框数占比')]
index=next(i for i,b in enumerate(blocks) if b.get('tableId') and next(t for t in tables if t['id']==b['tableId'])['title']=='640等效短边分布')
blocks.insert(index,dict(id='scale_chart',type='chart',chartId='scales'))
(D/'artifact.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
payload['snapshot']['datasets']['scales']=[{'size':r['size'],'group':label,'share':r[k]} for r in rows for k,label in [('original_train','原始train'),('combined_train','combined train'),('original_test','test')]]
c=payload['manifest']['charts'][0]; c.pop('xField'); c.pop('series'); c['encodings']={'x':{'field':'size','type':'nominal'},'y':{'field':'share','type':'quantitative','format':'percent'},'color':{'field':'group','type':'nominal'}}
(D/'artifact.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
sources[0]['query']['language']='python'
sources[0]['query']['query']='short = np.minimum(boxes[:, 3], boxes[:, 4]) * 640; counts = np.histogram(short, [0, 8, 16, 32, np.inf])[0]; shares = counts / len(boxes)'
sources[1]['query']['language']='manual transcription'
sources[1]['query']['query']='Read eval_metric/map, map_50, map_75, map_small, map_medium, map_large, mar_small from user-provided metrics_summary.json screenshots.'
(D/'artifact.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
import sqlite3
conn=sqlite3.connect(':memory:')
conn.execute('CREATE TABLE audited_scales(size TEXT, cohort TEXT, share REAL)')
conn.executemany('INSERT INTO audited_scales VALUES (?, ?, ?)',[(r['size'],r['group'],r['share']) for r in payload['snapshot']['datasets']['scales']])
sql='SELECT size, cohort AS "group", share FROM audited_scales'
conn.row_factory=sqlite3.Row
payload['snapshot']['datasets']['scales']=[dict(r) for r in conn.execute(sql)]
sources[0]['query'].update(engine='SQLite over Python audit results',language='sql',sql=sql,tables_used=['audited_scales (audit_results.json derived size shares)'])
# Table evidence is transcribed and materialized before selection for the report.
conn.execute('CREATE TABLE screenshot_metrics(row_json TEXT)')
for table in tables:
 if table['sourceId']=='screenshots':conn.executemany('INSERT INTO screenshot_metrics VALUES (?)',[(json.dumps(r,ensure_ascii=False),) for r in datasets[table['dataset']]])
list(conn.execute('SELECT row_json FROM screenshot_metrics'))
sources[1]['query'].update(engine='SQLite over screenshot transcription',language='sql',sql='SELECT row_json FROM screenshot_metrics',tables_used=['screenshot_metrics (user screenshot transcription)'])
(D/'artifact.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
# Keep long path references readable at narrow widths. Exact paths remain in source metadata and audit files.
for b in blocks:
 if b['type']=='markdown':
  b['body']=b['body'].replace('`conda run -n lightlytrain python reports/face_wider_combined_audit/audit.py`','以下命令（从仓库根目录运行）：\n\n```bash\nconda run -n lightlytrain python \\\n  reports/face_wider_combined_audit/audit.py\n```\n')
  # Natural break opportunities in long technical identifiers for the reader.
  b['body']=re.sub(r'(?<![A-Za-z0-9])([A-Za-z0-9_./-]{45,})(?![A-Za-z0-9])',lambda m:m.group(0).replace('_','_\u200b').replace('/','/\u200b') if 'http' not in m.group(0) else m.group(0),b['body'])
(D/'artifact.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
# Work around the portable reader's 100vw sticky header overflowing with classic scrollbars.
(D/'artifact.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
# Each native table gets the exact selection used to materialize its reviewed rows.
import copy
for table in tables:
 tname=table['id']; data=datasets[table['dataset']]; keys=list(data[0]); conn.execute('CREATE TABLE '+tname+' ('+', '.join(k+' TEXT' for k in keys)+')')
 conn.executemany('INSERT INTO '+tname+' VALUES ('+','.join('?' for _ in keys)+')',[[r[k] for k in keys] for r in data])
 q='SELECT '+', '.join(keys)+' FROM '+tname
 payload['snapshot']['datasets'][table['dataset']]=[dict(r) for r in conn.execute(q)]
 source=copy.deepcopy(next(s for s in sources if s['id']==table['sourceId'])); source['id']='source_'+tname
 source['query'].update(sql=q,query=q,tables_used=[tname+' (reviewed report rows)'])
 sources.append(source); table['sourceId']=source['id']
(D/'artifact.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
