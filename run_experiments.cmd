@REM run all *.yaml under job folder if hasn't been run once
@ECHO OFF

python run.py ^
  -D data\test.csv
  -X job ^
  --name test ^
  --no_overwrite
