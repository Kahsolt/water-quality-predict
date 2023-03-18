@ECHO OFF

SET PYTHONPATH=%CD%


ECHO ^>^> test descriptor ...
python modules\descriptor.py
IF ERRORLEVEL 1 GOTO ERROR
ECHO.

ECHO ^>^> test preprocess ...
python modules\preprocess.py
IF ERRORLEVEL 1 GOTO ERROR
ECHO.

ECHO ^>^> test dataset ...
python modules\dataset.py
IF ERRORLEVEL 1 GOTO ERROR
ECHO.

ECHO ^>^> test transform ...
python modules\transform.py
IF ERRORLEVEL 1 GOTO ERROR
ECHO.

GOTO PASS

:ERROR
ECHO ^<^< errorlevel %ERRORLEVEL%


:PASS
