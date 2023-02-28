#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/28 

import os
from pathlib import Path
from traceback import format_exc

from flask import Flask, request

import run as RT

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log'

app = Flask(__name__)
cur_expname = None


@app.route('/list')
def index():
  return [exp.name for exp in LOG_PATH.iterdir() if exp.is_dir()]


@app.route('/predict')
def hana_tap():
  req = request.get_json()
  print(f'>> req: {req}')

  try:
    expname = req['expname']
    x = req['x']

    if expname != cur_expname:
      RT.env = { }
      RT.job = RT.load_job(LOG_PATH / expname / 'job.yaml')
      manager = RT.env['manager']
      model = RT.env['model']
      cur_expname = expname

    y = manager.infer(model, x)
    return {'y': y.tolist()}
  except:
    return {'error': format_exc()}


@app.route('/debug')
def debug():
  s = ''
  s += f'<p>os.getcwd(): {os.getcwd()}</p>'
  s += f'<p>BASE_PATH: {BASE_PATH}</p>'
  s += f'<p>LOG_PATH: {LOG_PATH}</p>'
  return s


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
