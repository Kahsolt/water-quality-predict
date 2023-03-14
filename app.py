#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/28 

import os
import yaml
from pathlib import Path
from traceback import format_exc

import numpy as np
from flask import Flask, request, jsonify, render_template

from modules.util import seed_everything
import run as RT

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data'
LOG_PATH = BASE_PATH / 'log'

app = Flask(__name__, template_folder='.')
cur_job: str = None


@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')


@app.route('/job/list', methods=['GET'])
def job_list():
  try:
    return jsonify([job.name for job in LOG_PATH.iterdir() if job.is_dir()])
  except:
    return jsonify({'error': format_exc()})


@app.route('/job/<name>', methods=['GET'])
def job_(name):
  try:
    with open(LOG_PATH / name / 'job.yaml', 'r', encoding='utf-8') as fh:
      config = yaml.safe_load(fh)
    return jsonify(config)
  except:
    return jsonify({'error': format_exc()})


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


@app.route('/infer', methods=['POST'])
def infer():
  global cur_job

  try:
    req = request.get_json()
    job = req['job']
    x = np.asarray(req['x'], dtype=np.float32)
    print(f'>> job: {job}')
    print(f'>> x.shape: {x.shape}')

    if job != cur_job:
      # init job
      RT.job = RT.load_job(LOG_PATH / job / 'job.yaml')
      RT.env.clear()
      RT.env['log_dp'] = LOG_PATH / job
      seed_everything(RT.job_get('misc/seed'))

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
  s = ''
  s += f'<p>pwd: {os.getcwd()}</p>'
  s += f'<p>BASE_PATH: {BASE_PATH}</p>'
  s += f'<p>LOG_PATH: {LOG_PATH}</p>'
  return s


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
