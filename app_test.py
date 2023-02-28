#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/28 

from traceback import print_exc
import numpy as np
import requests as R

API_BASE = 'http://localhost:5000'
GET  = lambda api: R.get (API_BASE + api)
POST = lambda api, json=None: R.post(API_BASE + api, json=json)


def test_index():
  r = GET('/')
  assert r.status_code == 200


def test_job():
  r = GET('/job/list')
  assert r.status_code == 200

  job = r.json()[0]
  r = GET(f'/job/{job}')
  assert r.status_code == 200


def test_infer():
  json = {
    'job': 'XGBoost_rgr_h_overlap',
    'x': np.random.uniform(size=[168, 1]).tolist(),
  }
  r = POST('/infer', json)
  assert r.status_code == 200
  y = np.asarray(r.json()['y'])
  print('y.shape:', y.shape)

  json = {
    'job': 'XGBoost_clf_3h',
    'x': np.random.uniform(size=[24, 1]).tolist(),
  }
  r = POST('/infer', json)
  assert r.status_code == 200
  y = np.asarray(r.json()['y'])
  print('y.shape:', y.shape)


def test_debug():
  r = GET('/debug')
  assert r.status_code == 200


if __name__ == '__main__':
  testers = [
    test_index,
    test_job,
    test_infer,
    test_debug,
  ]

  for tester in testers:
    try:
      print(f'>> Testing {tester.__name__}...')
      tester()
    except:
      print_exc()
