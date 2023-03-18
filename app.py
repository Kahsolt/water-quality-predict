#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/28 

import os
import yaml
from pathlib import Path
from threading import Thread
from traceback import format_exc

import numpy as np
from flask import Flask, request
from flask import redirect, jsonify, render_template, send_file

from modules.descriptor import Descriptor
from modules.util import seed_everything
from modules.typing import *

import run as RT
from config import *


app = Flask(__name__, template_folder=HTML_PATH)
envs: Dict[str, Env] = { }
cur_job: Descriptor = None


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
  return jsonify([job.stem for job in (BASE_PATH / 'modules' / 'models').iterdir()
                  if job.suffix == '.py' and job.stem != '___init___'])


@app.route('/job')
def job():
  return jsonify([job.stem for job in JOB_PATH.iterdir() if job.suffix == '.yaml'])


@app.route('/job/<name>', methods=['GET', 'POST', 'DELETE'])
def job_(name:str):
  job_file = JOB_PATH / f'{name}.yaml'

  if request.method == 'GET':
    if not job_file.exists(): return jsonify({'ok': False, 'error': 'job file not found'})

    format = request.args.get('format', 'yaml')
    if format == 'yaml':
      return send_file(job_file, mimetype='text/yaml')
    elif format == 'json':
      with open(job_file, 'r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh)
      return jsonify({'ok': True, 'job': config})
    else:
      return jsonify({'ok': False, 'error': f'unknown format: {format}'})

  elif request.method == 'POST':
    if len(request.files) != 1:
      return jsonify({'ok': False, 'error': f'only need 1 file, but got {len(request.files)}'})
    file = request.files[0]

  elif request.method == 'DELETE':
    if not job_file.exists(): return jsonify({'ok': False, 'error': 'file not fould'})

    job_file.unlink()
    return jsonify({'ok': True})

  else: return jsonify({'ok': False, 'error': f'unspported method {request.method}'})


@app.route('/task', methods=['GET', 'POST'])
def task():
  if request.method == 'GET':
    return jsonify([task.name for task in LOG_PATH.iterdir() if task.is_dir()])

  elif request.method == 'POST':
    try:
      if len(request.files) != 1:
        return jsonify({'ok': False, 'error': f'only need 1 file, but got {len(request.files)}'})
      file = request.files[0]
      data = file.stream.read()
      fn = Path(file.filename).stem
      print('fn:', fn)

      data: Dict = request.json
      name = data.get('name', fn)
      jobs = data.get('jobs')

    except: return jsonify({'ok': False, 'error': format_exc()})

  else: return jsonify({'ok': False, 'error': f'unspported method {request.method}'})


@app.route('/task/<name>', methods=['GET', 'POST', 'DELETE'])
def task_(name:str):
  task_folder = LOG_PATH / name
  if not task_folder.is_dir(): return jsonify({'ok': False, 'error': 'task folder not found'})

  data = request.json

  if request.method == 'GET':
    pass

  elif request.method == 'POST':
    pass

  elif request.method == 'DELETE':
    pass

  else: return jsonify({'ok': False, 'error': f'unspported method {request.method}'})


@app.route('/infer/<task>/<job>', methods=['POST'])
def infer_(task:str, job:str):
  fullname = f'{task}-{job}'
  job_folder = LOG_PATH / task / job
  if not job_folder.is_dir(): return jsonify({'ok': False, 'error': 'job folder not found'})

  req = request.get_json()
  x = np.asarray(req['x'], dtype=np.float32)
  print(f'>> task: {task}')
  print(f'>> job: {job}')
  print(f'>> x.shape: {x.shape}')

  if fullname not in envs:
    envs[fullname] = RT.create_env(task, job, is_pretrained=True)

  env = envs[fullname]
  manager = env['manager']
  model = env['model']
  
  y = manager.infer(model, x, Logger)
  return jsonify({'y': y.tolist()})


if __name__ == '__main__':
  @app.route('/')
  def index():
    return redirect('/doc/api')

  app.run(host='0.0.0.0', debug=True)
