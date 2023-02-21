@REM run all *.yaml under job folder if hasn't been run once
@ECHO OFF

FOR /R %%f in (job\*.yaml) DO (
  ECHO [run] python run.py -J %%f
  python run.py -J %%f --no_overwrite
)
