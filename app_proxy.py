#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/28 

from app import *


@app.route('/page2/getFittingCurve')
def getFittingCurve():
  return 'getFittingCurve'


@app.route('/page2/get6hPredictionResult')
def get6hPredictionResult():
  return 'get6hPredictionResult'


@app.route('/page2/getModelPerformance')
def getModelPerformance():
  return 'getModelPerformance'


@app.route('/page2/getExceedingPredictionResult')
def getExceedingPredictionResult():
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
