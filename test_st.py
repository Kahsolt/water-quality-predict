#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/20 

# server integral test

import os
import random
from time import sleep
from traceback import print_exc

import requests as R
from requests import Response

from modules.utils import *

HOST = os.environ.get('HOST', 'localhost')
PORT = os.environ.get('PORT', 5000)
API_BASE = f'http://{HOST}:{PORT}'
EP = lambda api: f'{API_BASE}{api}'

task_name = 'test-api'


def test_case(fn:Callable):
  def wrapper(*args, **kwargs):
    print(f'>> execute {fn.__name__}')
    try:
      fn(*args, **kwargs)
    except:
      print_exc()
      exit(-1)
  return wrapper


@test_case
def test_basic_info():
  resp: Response = R.get(EP('/model'))
  assert resp.ok, resp ; r = resp.json() ; print(r)

  resp: Response = R.get(EP('/job'))
  assert resp.ok, resp ; r = resp.json() ; print(r)
  job_name = random.choice(r['data']['jobs'])

  resp: Response = R.get(EP(f'/job/{job_name}?format=json'))
  assert resp.ok, resp ; r = resp.json() ; print(r)

  job_file = JOB_PATH / f'{job_name}.yaml'
  new_job_name = 'rgr_Test'
  resp: Response = R.post(
    EP(f'/job/{new_job_name}'), 
    files={new_job_name: open(str(job_file), 'rb')}, 
  )
  assert resp.ok, resp ; r = resp.json() ; print(r)

  resp: Response = R.delete(EP(f'/job/{new_job_name}'))
  assert resp.ok, resp ; r = resp.json() ; print(r)

  resp: Response = R.get(EP('/debug'))
  assert resp.ok, resp ; r = resp.text ; print(r)

  resp: Response = R.get(EP('/task'))
  assert resp.ok, resp ; r = resp.json() ; print(r)

  resp: Response = R.get(EP('/runtime'))
  assert resp.ok, resp ; r = resp.json() ; print(r)


@test_case
def test_train_routine():
  ''' Step 0: delete old task '''
  resp = R.delete(EP(f'/task/{task_name}'))
  assert resp.ok, resp ; r = resp.json() ; print(r)

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
  assert resp.ok, resp ; r = resp.json() ; print(r)
  assert task_name == r['data']['name']

  ''' Step 2: wait for task finish '''
  resp = R.get(EP('/runtime'))
  assert resp.ok, resp ; r = resp.json() ; print(r)

  resp = R.get(EP('/task'))
  assert resp.ok, resp ; r = resp.json() ; print(r)

  finished = False
  while not finished:
    resp = R.get(EP(f'/task/{task_name}'))
    assert resp.ok, resp ; r = resp.json() ; print(r)

    status = Status(r['data']['status'])
    if status == Status.FAILED:
      print('!! Task Failed !!')
      return

    finished = status == Status.FINISHED
    jobs: dict = r['data']['jobs']

    sleep(5)

  ''' Step 3: download log '''
  job_name = random.choice(list(jobs.keys()))
  resp = R.get(EP(f'/log/{task_name}/{job_name}.log'))
  assert resp.ok, resp.text


@test_case
def test_infer_routine():
  resp = R.get(EP(f'/task/{task_name}'))
  assert resp.ok, resp
  job_states = resp.json()['data']['jobs']
  #print(job_states)

  # [T=256, D=1], any length is ok
  data = np.random.uniform(size=[256, 1]).astype(np.float32)
  data = ndarray_to_list(data)
  ts = ts_now()
  T = [ts + i * 3600 for i in range(len(data))]

  jobs: dict = resp.json()['data']['jobs']
  for job in jobs.keys():
    print(f'>> querying {job!r}...')
    if job not in job_states or job_states[job]['status'] != 'finished': continue

    resp = R.post(
      EP(f'/infer/{task_name}/{job}'),
      json={'data': data, 'time': T, 'roll': 6 if job.startswith('rgr_') else 1},
    )
    assert resp.ok, resp ; r = resp.json()

    if not r['ok']: breakpoint() ; continue

    pred = list_to_ndarray(r['data']['pred'])
    print('pred.shape:', pred.shape)
    print('pred.mean:', pred.mean())
    if 'prob' in r['data']:
      prob = list_to_ndarray(r['data']['prob'])
      print('prob.shape:', prob.shape)
      print('prob.mean:', prob.mean())


@test_case
def test_infer_inplace_routine():
  resp = R.get(EP(f'/task/{task_name}'))
  assert resp.ok, resp
  job_states = resp.json()['data']['jobs']
  #print(job_states)

  jobs: dict = resp.json()['data']['jobs']
  for job in jobs.keys():
    print(f'>> querying {job!r}...')
    if job not in job_states or job_states[job]['status'] != 'finished': continue

    resp = R.post(
      EP(f'/infer/{task_name}/{job}'),
      json={'inplace': True},
    )
    assert resp.ok, resp ; r = resp.json()

    if not r['ok']: breakpoint()

    pred = list_to_ndarray(r['data']['pred'])
    print('pred.shape:', pred.shape)
    print('pred.mean:', pred.mean())
    if 'prob' in r['data']:
      prob = list_to_ndarray(r['data']['prob'])
      print('prob.mean:', prob.mean())
    if 'lbl' in r['data']:
      lbl = list_to_ndarray(r['data']['lbl'])
      print(f'lbl bad ratio: {(lbl > 0).sum() / len(lbl):.3%}')


@test_case
def test_train_pressure(idx:int):
  real_task_name = f'{task_name}-{idx}'

  ''' Step 0: delete old task '''
  resp = R.delete(EP(f'/task/{real_task_name}'))
  assert resp.ok, resp ; r = resp.json() ; print(r)

  ''' Step 1: create task '''
  resp: Response = R.post(
    EP('/task'), 
    files={
      'json': json.dumps({
        'target': ['data', 'train', 'eval', 'infer'],
        #'target': 'all',
        'jobs': None,
        'name': real_task_name,
      }),
      'csv': open('data/test.csv', 'rb'),
    },
  )
  assert resp.ok, resp ; r = resp.json() ; print(r)
  assert task_name == r['data']['name']


if __name__ == '__main__':
  test_basic_info()
  test_train_routine()
  test_infer_inplace_routine()
  test_infer_routine()

  for i in range(10):
    test_train_pressure(i)
