#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/20 

import random
from time import sleep
import requests as R
from requests import Response

from config import *
from modules.util import *

API_BASE = 'http://localhost:5000'
EP = lambda api: f'{API_BASE}{api}'

task_name = 'test-api'


def test_basic_info():
  resp: Response = R.get(EP('/model'))
  assert resp.ok ; r = resp.json() ; print(r)

  resp: Response = R.get(EP('/job'))
  assert resp.ok ; r = resp.json() ; print(r)
  job_name = random.choice(r['data']['jobs'])

  resp: Response = R.get(EP(f'/job/{job_name}?format=json'))
  assert resp.ok ; r = resp.json() ; print(r)

  job_file = JOB_PATH / f'{job_name}.yaml'
  new_job_name = 'rgr_Test'
  resp: Response = R.post(
    EP(f'/job/{new_job_name}'), 
    files={new_job_name: open(str(job_file), 'rb')}, 
  )
  assert resp.ok ; r = resp.json() ; print(r)

  resp: Response = R.delete(EP(f'/job/{new_job_name}'))
  assert resp.ok ; r = resp.json() ; print(r)

  resp: Response = R.get(EP('/debug'))
  assert resp.ok ; r = resp.text ; print(r)

  resp: Response = R.get(EP('/task'))
  assert resp.ok ; r = resp.json() ; print(r)

  resp: Response = R.get(EP('/runtime'))
  assert resp.ok ; r = resp.json() ; print(r)


def test_train_routine():
  ''' Step 0: delete old task '''
  resp = R.delete(EP(f'/task/{task_name}'))
  assert resp.ok ; r = resp.json() ; print(r)

  ''' Step 1: create task '''
  resp: Response = R.post(
    EP('/task'), 
    files={
      'json': json.dumps({
        'target': ['data', 'train', 'eval', 'infer'],
        #'target': 'all',
        'jobs': None,
        'name': task_name,
      }),
      'csv': open('data/test.csv', 'rb'),
    },
  )
  assert resp.ok ; r = resp.json() ; print(r)
  assert task_name == r['data']['name']

  ''' Step 2: wait for task finish '''
  resp = R.get(EP('/runtime'))
  assert resp.ok ; r = resp.json() ; print(r)

  resp = R.get(EP('/task'))
  assert resp.ok ; r = resp.json() ; print(r)

  finished = False
  while not finished:
    resp = R.get(EP(f'/task/{task_name}'))
    assert resp.ok ; r = resp.json() ; print(r)

    status = Status(r['data']['status'])
    if status == Status.FAILED:
      print('!! Task Failed !!')
      return

    finished = status == Status.FINISHED
    jobs: dict = r['data']['jobs']

    sleep(5)

  ''' Step 3: download log '''
  job_name = random.choice(list(jobs.keys()))
  resp = R.get(EP(f'/task/{task_name}/{job_name}.log'))
  assert resp.ok, resp.text


def test_infer_routine():
  resp = R.get(EP(f'/task/{task_name}'))
  assert resp.ok

  # [T=256, D=1], any length is ok
  data = np.random.uniform(size=[256, 1]).astype(np.float32)
  base64, shape = ndarray_to_base64(data)

  jobs: dict = resp.json()['data']['jobs']
  for job in jobs.keys():
    print(f'>> querying {job!r}...')

    resp = R.post(
      EP(f'/infer/{task_name}/{job}'),
      json={'data': base64, 'shape': shape},
    )
    assert resp.ok ; r = resp.json()

    pred = base64_to_ndarray(r['data']['pred'], r['data']['shape'])
    print(pred)


if __name__ == '__main__':
  #test_basic_info()
  #test_train_routine()
  test_infer_routine()
