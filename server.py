#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/28 

# server (webapp)

import warnings ; warnings.filterwarnings('ignore', category=UserWarning)

import os
import shutil
import psutil
from argparse import ArgumentParser
from traceback import print_exc, format_exc
import logging
import gc

from flask import Flask, request, Response
from flask import redirect, jsonify, render_template, send_file

from modules.trainer import Trainer, TaskInit, RunMeta
from modules.predictor import predict_from_request
from modules.utils import *

app = Flask(__name__, template_folder=HTML_PATH)


def resp_ok(data:Union[dict, list]=None) -> Response:
  return jsonify({
    'ok': True,
    'data': data,
  })

def resp_error(err:str) -> Response:
  gc.collect()
  return jsonify({
    'ok': False,
    'error': err,
  })


@app.route('/', methods=['GET'])
def index():
  return redirect('/doc/api')


@app.route('/doc/<page>', methods=['GET'])
def doc_(page:str):
  return render_template(f'{page}.html')


@app.route('/debug', methods=['GET'])
def debug():
  pid = os.getpid()
  loadavg = psutil.getloadavg()
  p = psutil.Process(pid)
  meminfo = p.memory_full_info()
  
  return f'''
      <div>
        <p>pwd: {os.getcwd()}</p>
        <p>BASE_PATH: {BASE_PATH}</p>
        <p>HTML_PATH: {HTML_PATH}</p>
        <p>JOB_PATH: {JOB_PATH}</p>
        <p>LOG_PATH: {LOG_PATH}</p>
        <p>loadavg: {loadavg}</p>
        <p>mem use: {meminfo.rss / 2**20:.3f} MB</p>
        <p>mem vms: {meminfo.vms / 2**20:.3f} MB</p>
        <p>mem percent: {p.memory_percent()} %</p>
      </div>
    '''


@app.route('/model', methods=['GET'])
def model():
  models = [fp.stem for fp in MODEL_PATH.iterdir() if fp.suffix == '.py' and fp.stem != '___init___']
  return resp_ok({'models': models})


@app.route('/job', methods=['GET'])
def job():
  jobs = [fp.stem for fp in JOB_PATH.iterdir() if fp.suffix == '.yaml']
  return resp_ok({'jobs': jobs})


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

    file.save(job_file) ; file.close()
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
    req: Dict = json.loads(fjson) if fjson is not None else {}
    if 'name' in req and Path(LOG_PATH / req['name']).exists(): return resp_error('task already exists, should use POST /task/<name> to retrain/modify')

    name = req.get('name', datetime_str())
    target = fix_targets(req.get('target'))
    if not len(target): return resp_error('bad target')
    jobs = fix_jobs(req.get('jobs'))
    if not len(jobs): return resp_error('bad jobs')
    thresh: float = req.get('thresh')

    fcsv = request.files['csv'].stream.read()
    task_init = TaskInit(
      name=name,
      data=fcsv,
      target=target,
      jobs=jobs,
      thresh=thresh,
    )
    init_fp = TMP_PATH / f'{name}-{rand_str(4)}.pkl'
    save_pickle(task_init, init_fp)
    trainer.add_task(name, init_fp)

    for f in request.files.values(): f.close()
    return resp_ok({'name': name})

  elif request.method == 'GET':
    return resp_ok({'tasks': [task.name for task in LOG_PATH.iterdir() if task.is_dir()]})


@app.route('/task/<name>', methods=['POST', 'GET', 'DELETE'])
def task_(name:str):
  task_folder = LOG_PATH / name
  if not task_folder.exists(): return resp_error('task folder not exists')

  if request.method == 'POST':
    if len(request.files) > 1: return resp_error(f'only need 0 or 1 file, but got {len(request.files)}')

    try:
      req: Dict = request.json

      target = fix_targets(req.get('target'))
      if not len(target): return resp_error('bad target')
      jobs = fix_jobs(req.get('jobs'))
      if not len(jobs): return resp_error('bad jobs')
      data = request.files[0].stream.read() if len(request.files) else None
      thresh: float = req.get('thresh')
      task_init = TaskInit(
        name=name,
        data=data,
        target=target,
        jobs=jobs,
        thresh=thresh,
      )
      init_fp = TMP_PATH / f'{name}-{rand_str(4)}.pkl'
      save_pickle(task_init, init_fp)

      for dp in task_folder.iterdir():
        if dp.is_dir():
          shutil.rmtree(str(dp))

      trainer.add_task(name, init_fp)

      for f in request.files.values(): f.close()
      return resp_ok()
    except:
      return resp_error(format_exc())

  elif request.method == 'GET':
    return resp_ok(load_json(task_folder / TASK_FILE)[-1])    # only last run

  elif request.method == 'DELETE':
    shutil.rmtree(str(task_folder))
    return resp_ok()


@app.route('/infer/<task>/<job>', methods=['POST'])
def infer_(task:str, job:str):
  job_folder = LOG_PATH / task / job
  if not job_folder.exists(): return resp_error('job folder not exists')

  try:
    req: Dict = request.json

    if req.get('inplace', False):
      time: Time = load_pickle(job_folder / TIME_FILE)
      seq:  Seq  = load_pickle(job_folder / PREPROCESS_FILE)
      lbl:  Seq  = load_pickle(job_folder / LABEL_FILE)
      pred: Seq  = load_pickle(job_folder / PREDICT_FILE)[0]    # only need pred_o
      time = time.to_list()
      seq  = ndarray_to_list(seq)
      pred = ndarray_to_list(pred)
      r = {
        'time': time,
        'seq':  seq, 
        'pred': pred, 
      }
      if lbl is not None:
        r.update({'lbl': ndarray_to_list(lbl)})
      return resp_ok(r)
    else:
      job_file = job_folder / JOB_FILE

      t: Frame = list_to_ndarray(req['time']).astype(np.int32) if 'time' in req else None
      x: Frame = list_to_ndarray(req['data'])
      roll: int = req.get('roll', 1)
      if job.startswith('rgr_'):
        y = predict_from_request(job_file, x, t, roll=roll, prob=False)
        pred = ndarray_to_list(y)
        return resp_ok({'pred': pred})
      else:
        y, y_prb = predict_from_request(job_file, x, t, roll=roll, prob=True)
        pred = ndarray_to_list(y)
        prob = ndarray_to_list(y_prb)
        return resp_ok({'pred': pred, 'prob': prob})

  except:
    return resp_error(format_exc())


@app.route('/runtime', methods=['GET'])
def runtime():
  filters_str = ','.join([e.value for e in [Status.QUEUING, Status.RUNNING]])
  try: filters_str = request.args.get('status', filters_str)
  except: pass
  filters = filters_str.split(',')

  if 'all' in filters:
    runtime_hist = trainer.run_meta_list
  else:
    filtered: List[RunMeta] = []
    for run_meta in trainer.run_meta_list:
      if run_meta.status.value in filters:
        filtered.append(run_meta)
    runtime_hist = filtered
  runtime_hist = RunMeta.to_dict_list(runtime_hist)
  return resp_ok({'runtime_hist': serialize_json(runtime_hist)})


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

  zip_fp = TMP_PATH / f'{get_fullname(task, job)}.zip'
  make_zip(job_folder, zip_fp)
  return send_file(zip_fp, mimetype='application/zip')


@app.route('/log/<task>/<job>.log', methods=['GET'])
def log__log(task:str, job:str):
  job_folder = LOG_PATH / task / job
  if not job_folder.exists(): return resp_error('job folder not exists')

  logs = read_txt(job_folder / LOG_FILE)
  return '<br/>'.join(logs)


@app.route('/log/clean', methods=['GET'])
def log_clean():
  def walk(dp:Path):
    for fp in dp.iterdir():
      if fp.is_dir():
        walk(fp)
      else:
        if fp.suffix == '.png':
          print(f'>> delete {fp.relative_to(BASE_PATH)}')
          fp.unlink()
  walk(LOG_PATH)
  return resp_ok()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-H', '--host', type=str, default='0.0.0.0')
  parser.add_argument('-P', '--port', type=int, default=5000)
  parser.add_argument('--no_trainer', action='store_true')
  parser.add_argument('--n_workers', type=int, default=1)
  parser.add_argument('--queue_timeout', type=int, default=5)
  args = parser.parse_args()

  has_trainer = not args.no_trainer

  if has_trainer:
    trainer = Trainer(args.n_workers, args.queue_timeout)
  try:
    if has_trainer: trainer.start()
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
  except KeyboardInterrupt:
    print('Exit by Ctrl+C')
  except:
    print_exc()
  finally:
    if has_trainer: trainer.stop()
    logging.shutdown()
