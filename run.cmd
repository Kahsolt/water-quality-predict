@ECHO OFF

REM run all jobs in 'job/*.yaml' over test dataset 'test.csv', if hasn't been run once

python run.py ^
  -D data\test.csv ^
  -X job ^
  --name test ^
  --no_overwrite
