#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/28 

import os
import shutil

from flask import Flask, request, Response
from flask import redirect, jsonify, render_template, send_file

from modules.typing import *
from modules.util import *

from config import *
from run import *
from worker import *

app = Flask(__name__, template_folder=HTML_PATH)


def resp_ok(data:Union[dict, list]=None) -> Response:
  return jsonify({
    'ok': True,
    'data': data,
  })

def resp_error(errmsg:str) -> Response:
  return jsonify({
    'ok': False,
    'error': errmsg,
  })


def fix_target(target) -> List[str]:
  if not target: return ['all']
  if isinstance(target, str): target = [target]
  valid_tgt = enum_values(Target)
  for tgt in target: assert tgt in valid_tgt
  return target

def fix_jobs(jobs) -> List[str]:
  valid_job = [job.stem for job in JOB_PATH.iterdir() if job.suffix == '.yaml']
  if not jobs: return valid_job
  for job in jobs: assert job in valid_job
  return jobs


@app.route('/doc/<page>')
def doc_(page:str):
  return render_template(f'{page}.html')


@app.route('/debug')
def debug():
  return f'''
<div>
  <p>pwd: {os.getcwd()}</p>
  <p>BASE_PATH: {BASE_PATH}</p>
  <p>HTML_PATH: {HTML_PATH}</p>
  <p>JOB_PATH: {JOB_PATH}</p>
  <p>LOG_PATH: {LOG_PATH}</p>
</div>
'''


@app.route('/model')
def model():
  return resp_ok({'models': [job.stem for job in (BASE_PATH / 'modules' / 'models').iterdir() 
                                      if job.suffix == '.py' and job.stem != '___init___']})


@app.route('/job')
def job():
  return resp_ok({'jobs': [job.stem for job in JOB_PATH.iterdir() if job.suffix == '.yaml']})


@app.route('/job/<name>', methods=['POST', 'GET', 'DELETE'])
def job_(name:str):
  job_file = JOB_PATH / f'{name}.yaml'

  if request.method in ['GET', 'DELETE']:
    if not job_file.exists(): return resp_error('job file not exists')

  if request.method == 'POST':
    if len(request.files) != 1: return resp_error(f'only need 1 file, but got {len(request.files)} files')
    name = list(request.files.keys())[0]
    file = request.files[name]

    if name.split('_')[0] not in enum_values(TaskType): return resp_error('file name must starts with "rgr_" or "clf_"')

    overwrite = 0
    try: overwrite = int(request.args.get('overwrite', overwrite))
    except: pass
    if not overwrite and job_file.exists(): return resp_error('no overwrite existing file')

    file.save(job_file)
    return resp_ok()

  elif request.method == 'GET':
    format = request.args.get('format', 'yaml')
    if format == 'yaml':
      return send_file(job_file, mimetype='text/yaml')
    elif format == 'json':
      return resp_ok({'job': load_yaml(job_file)})
    else:
      return resp_error(f'unknown format: {format}')

  elif request.method == 'DELETE':
    job_file.unlink()
    return resp_ok()


@app.route('/task', methods=['POST', 'GET'])
def task():
  if request.method == 'POST':
    if len(request.files) > 2: return resp_error(f'only need at most 2 files: "json" ad "csv", but got {len(request.files)}')
    if 'csv' not in request.files: return resp_error(f'not found required "csv" in uploaded files')

    fjson = request.files['json'].stream.read() if 'json' in request.files else None
    fcsv  = request.files['csv'] .stream.read()

    req: Dict = json.loads(fjson) if fjson is not None else {}
    if 'name' in req and (LOG_PATH / req['name']).exists(): return resp_error('task already exists, should use POST /task/<name> to retrain/modify')

    name = req.get('name', timestr())
    target = fix_target(req.get('target'))
    jobs = fix_jobs(req.get('jobs'))
    task_init: TaskInit = {
      'name': name,
      'data': fcsv,
      'target': target,
      'jobs': jobs,
    }
    TMP_PATH.mkdir(exist_ok=True, parents=True)
    init_fp = TMP_PATH / f'{name}-{rand_str(4)}.pkl'
    save_pickle(task_init, init_fp)
    trainer.add_task(name, init_fp)

    return resp_ok({'name': name})

  elif request.method == 'GET':
    return resp_ok({'tasks': [task.name for task in LOG_PATH.iterdir() if task.is_dir()]})


@app.route('/task/<name>', methods=['POST', 'GET', 'DELETE'])
def task_(name:str):
  task_folder = LOG_PATH / name
  if not task_folder.exists(): return resp_error('task folder not exists')

  if request.method == 'POST':
    if len(request.files) > 1: return resp_error(f'only need 0 or 1 file, but got {len(request.files)}')

    req: Dict = request.json
    data = request.files[0].stream.read() if len(request.files) else None
    target = fix_target(req.get('target'))
    jobs = fix_jobs(req.get('jobs'))
    task_init: TaskInit = {
      'name': name,
      'data': data,
      'target': target,
      'jobs': jobs,
    }
    TMP_PATH.mkdir(exist_ok=True, parents=True)
    init_fp = TMP_PATH / f'{name}-{rand_str(4)}.pkl'
    save_pickle(task_init, init_fp)
    trainer.add_task(name, init_fp)

    return resp_ok()

  elif request.method == 'GET':
    return resp_ok(load_json(task_folder / TASK_FILE)[-1])    # only last run

  elif request.method == 'DELETE':
    shutil.rmtree(str(task_folder))
    return resp_ok()


@app.route('/infer/<task>/<job>', methods=['POST'])
def infer_(task:str, job:str):
  job_folder = LOG_PATH / task / job
  if not job_folder.exists(): return resp_error('job folder not exists')

  req = request.json
  if req.get('inplace', False):
    try:
      seq:  Seq = load_pickle(job_folder / PREPROCESS_FILE)
      pred: Seq = load_pickle(job_folder / PREDICT_FILE)[0]    # only need pred_o
      return resp_ok({
        'seq': ndarray_to_bytes(seq), 
        'seq_shape': tuple(seq.shape),
        'pred': ndarray_to_bytes(pred), 
        'pred_shape': tuple(pred.shape),
      })
    except:
      return resp_error(format_exc())
  else:
    x: Frame = bytes_to_ndarray(req['data'], req['shape'])
    y = predictor.predict(task, job, x)
    return resp_error({'pred': ndarray_to_bytes(y), 'shape': tuple(y.shape)})


@app.route('/log/<task>', methods=['GET'])
def log_(task:str):
  task_folder = LOG_PATH / task
  if not task_folder.exists(): return resp_error('task folder not exists')

  zip_fp = TMP_PATH / f'{task}.zip'
  make_zip(task_folder,  zip_fp)
  return send_file(zip_fp, mimetype='application/zip')


@app.route('/log/<task>/<job>', methods=['GET'])
def log__(task:str, job:str):
  job_folder = LOG_PATH / task / job
  if not job_folder.exists(): return resp_error('job folder not exists')

  zip_fp = TMP_PATH / f'{task}@{job}.zip'
  make_zip(job_folder, zip_fp)
  return send_file(zip_fp, mimetype='application/zip')


@app.route('/log/<task>/<job>.log', methods=['GET'])
def log__log(task:str, job:str):
  job_folder = LOG_PATH / task / job
  if not job_folder.exists(): return resp_error('job folder not exists')

  with open(job_folder / LOG_FILE, 'r', encoding='utf-8') as fh:
    logs = fh.read().strip()
  return '<br/>'.join(logs.split('\n'))


@app.route('/merge_csv', methods=['POST'])
def merge_csv():
  return resp_ok({'status': 'not impl'})


@app.route('/runtime', methods=['GET'])
def runtime():
  filters = 'queuing,running'
  try: filters = request.args.get('status', filters)
  except: pass
  filters = filters.split(',')

  return resp_ok({'runtime_hist': [run for run in trainer.run_meta if run['status'] in filters]})


if __name__ == '__main__':
  @app.route('/')
  def index():
    return redirect('/doc/api')

  trainer = Trainer()
  predictor = Predictor()
  try:
    trainer.start()
    app.run(host='0.0.0.0', debug=True)
  finally:
    trainer.stop()
