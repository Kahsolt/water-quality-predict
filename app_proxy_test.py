#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/20 

import requests as R
from requests import Response

API_BASE = 'http://localhost:5000'

GET  = lambda api:       R.get (API_BASE + api)
POST = lambda api, data: R.post(API_BASE + api, json=data)


def test_getFittingCurve():
  data = {

  }
  resp: Response = POST('/page2/getFittingCurve', data)
  assert resp.status_code == 200


def test_get6hPredictionResult():
  data = {

  }
  resp: Response = POST('/page2/get6hPredictionResult', data)
  assert resp.status_code == 200


def test_getModelPerformance():
  data = {

  }
  resp: Response = POST('/page2/getModelPerformance', data)
  assert resp.status_code == 200


def test_getExceedingPredictionResult():
  data = {

  }
  resp: Response = POST('/page2/getExceedingPredictionResult', data)
  assert resp.status_code == 200


if __name__ == '__main__':
  test_getFittingCurve()
  test_get6hPredictionResult()
  test_getModelPerformance()
  test_getExceedingPredictionResult()
