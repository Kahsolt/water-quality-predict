#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/28 

from app import *


def resp_ok(data:Union[dict, list]=None) -> Response:
  return jsonify({
    'code': 200,
    'msg': 'success',
    'data': data,
  })

def resp_error(errmsg:str) -> Response:
  return jsonify({
    'code': 500,
    'msg': errmsg,
    'data': None,
  })


@app.route('/page2/getFittingCurve')
def getFittingCurve():
  req = request.json
  code   = req['dischargeCode']
  factor = req['monitorFactor']
  start  = req['start']
  end    = req['end']

  scores = {
    'R2': None,
    'MAE': None,
    'RMSE': None,
  }

  return resp_ok({
    'evaluation': scores,
    'realData': [],
    'fitData': [],
  })


@app.route('/page2/get6hPredictionResult')
def get6hPredictionResult():
  req = request.json
  code   = req['dischargeCode']
  factor = req['monitorFactor']
  start  = req['start']
  end    = req['end']

  return 'get6hPredictionResult'


@app.route('/page2/getModelPerformance')
def getModelPerformance():
  req = request.json
  code   = req['dischargeCode']
  factor = req['monitorFactor']
  start  = req['start']
  end    = req['end']

  return 'getModelPerformance'


@app.route('/page2/getExceedingPredictionResult')
def getExceedingPredictionResult():
  req = request.json
  code   = req['dischargeCode']
  factor = req['monitorFactor']
  start  = req['start']
  end    = req['end']

  return 'getExceedingPredictionResult'


if __name__ == '__main__':
  @app.route('/')
  def index():
    return redirect('/doc/api_proxy')

  trainer = Trainer()
  try:
    trainer.start()
    app.run(host='0.0.0.0', debug=True)
  finally:
    trainer.stop()
