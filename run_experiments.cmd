@REM run all *.yaml under job folder if hasn't been run once
@ECHO OFF

python run_batch.py -D job --no_overwrite
