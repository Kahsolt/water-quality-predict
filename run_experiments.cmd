@ECHO OFF

FOR /R %%f in (job\*.yaml) DO (
  ECHO [run] python run.py -J %%f
  python run.py -J %%f
)
