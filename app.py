#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/28 

import os
import yaml
from pathlib import Path
from threading import Thread
from typing import Dict
from traceback import format_exc

import numpy as np
from flask import Flask, request, jsonify, render_template, send_file

from modules.descriptor import Descriptor
from modules.util import seed_everything
import run as RT


BASE_PATH = Path(__file__).parent
JOB_PATH = BASE_PATH / 'job'
LOG_PATH = BASE_PATH / 'log'

app = Flask(__name__, template_folder='.')
cur_job: Descriptor = None


@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')


@app.route('/task', methods=['GET', 'POST'])
def task():
  try:
    if request.method == 'GET':
      return jsonify([task.name for task in LOG_PATH.iterdir() if task.is_dir()])

    elif request.method == 'POST':
      try:
        if len(request.files): return jsonify({'ok': False, 'error': f'file count should be exactly 1, but got {len(request.files)}'}) 

        file = request.files[0]
        fn = file.filename
        print('fn:', fn)

        data: Dict = request.json
        ttype = data['type']
        name = data.get('name', fn)
        jobs = data.get('jobs')

      except: return jsonify({'ok': False, 'error': format_exc()})

    else: return jsonify({'ok': False, 'error': f'unspported method {request.method}'})
  except: return jsonify({'ok': False, 'error': format_exc()})


@app.route('/task/<name>', methods=['GET', 'POST', 'DELETE'])
def task_(name:str):
  task_folder = LOG_PATH / name
  if not task_folder.is_dir(): return jsonify({'ok': False, 'error': 'task folder not found'})

  try: data = request.json
  except: return jsonify({'ok': False, 'error': format_exc()})

  try:
    if request.method == 'GET':
      pass

    elif request.method == 'POST':
      pass

    elif request.method == 'DELETE':
      pass

    else: return jsonify({'ok': False, 'error': f'unspported method {request.method}'})
  except: return jsonify({'ok': False, 'error': format_exc()})


@app.route('/job', methods=['GET'])
def job():
  try:
    if request.method == 'GET':
      return jsonify([job.stem for job in JOB_PATH.iterdir() if job.suffix == '.yaml'])
    
    else: return jsonify({'ok': False, 'error': f'unspported method {request.method}'})
  except: return jsonify({'ok': False, 'error': format_exc()})


@app.route('/job/<name>', methods=['GET', 'POST', 'DELETE'])
def job_(name:str):
  job_file = JOB_PATH / f'{name}.yaml'

  try:
    if request.method == 'GET':
      if not job_file.exists(): return jsonify({'ok': False, 'error': 'job file not found'})

      ftype = 'yaml'
      try: ftype = request.json.get('type', ftype)
      except: pass

      if ftype == 'yaml':
        return send_file(job_file, mimetype='text/yaml')
      elif ftype == 'json':
        with open(job_file, 'r', encoding='utf-8') as fh:
          config = yaml.safe_load(fh)
        return jsonify({'ok': True, 'job': config})
      else:
        return jsonify({'ok': False, 'error': f'unknown file type {ftype}'})

    elif request.method == 'POST':
      pass

    elif request.method == 'DELETE':
      if not job_file.exists(): return jsonify({'ok': False, 'error': 'file not fould'})

      job_file.unlink()
      return jsonify({'ok': True})

    else: return jsonify({'ok': False, 'error': f'unspported method {request.method}'})
  except: return jsonify({'ok': False, 'error': format_exc()})


@app.route('/infer/<task>', methods=['POST'])
def infer_(task:str):
  global cur_job

  try:
    req = request.get_json()
    job = req['job']
    x = np.asarray(req['x'], dtype=np.float32)
    print(f'>> task: {task}')
    print(f'>> job: {job}')
    print(f'>> x.shape: {x.shape}')

    if job != cur_job:
      # init job
      RT.job = Descriptor.load(LOG_PATH / task / job / 'job.yaml')
      RT.env.clear()
      RT.env['log_dp'] = LOG_PATH / task / job
      seed_everything(RT.job.get('seed'))

      # load job states
      @RT.require_model
      @RT.require_data
      def load_model_and_data():
        RT.env['model'] = RT.env['manager'].load(RT.env['model'], RT.env['log_dp'])
      load_model_and_data()

      # get context
      manager = RT.env['manager']
      model = RT.env['model']
      cur_job = job

    y = manager.infer(model, x)
    return jsonify({'y': y.tolist()})
  except:
    return jsonify({'error': format_exc()})


@app.route('/debug', methods=['GET'])
def debug():
  return f'''
<div>
  <p>pwd: {os.getcwd()}</p>
  <p>BASE_PATH: {BASE_PATH}</p>
  <p>JOB_PATH: {JOB_PATH}</p>
  <p>LOG_PATH: {LOG_PATH}</p>
</div>
'''


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)


'''
@app.route('/metric/list', methods=['GET'])
def metric_list():
  try:
    with open(LOG_PATH / 'metric-ranklist.txt', 'r', encoding='utf-8') as fh:
      data = fh.read().strip()
    ret = { }
    for sec in data.split('\n\n'):
      lines = sec.split('\n')
      title = lines[0][:-1]     # remove ':'
      ret[title] = { }
      for line in lines[1:]:
        name, score = line.split(':')
        ret[title][name] = float(score)
    return jsonify(ret)
  except:
    return jsonify({'error': format_exc()})


@app.route('/metric/<name>', methods=['GET'])
def metric_(name):
  try:
    with open(LOG_PATH / name / 'metric.txt', 'r', encoding='utf-8') as fh:
      data = fh.read().strip()
    ret = {}
    for line in data.split('\n'):
      name, score = line.split(':')
      ret[name] = float(score)
    return jsonify(ret)
  except:
    return jsonify({'error': format_exc()})
'''
