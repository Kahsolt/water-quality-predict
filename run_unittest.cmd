@REM run unit test for modules
@ECHO OFF


ECHO ^>^> test descriptor ...
python -m modules.descriptor
IF ERRORLEVEL 1 GOTO ERROR
ECHO.

ECHO ^>^> test preprocess ...
python -m modules.preprocess
IF ERRORLEVEL 1 GOTO ERROR
ECHO.

ECHO ^>^> test dataset ...
python -m modules.dataset
IF ERRORLEVEL 1 GOTO ERROR
ECHO.

ECHO ^>^> test transform ...
python -m modules.transform
IF ERRORLEVEL 1 GOTO ERROR
ECHO.

GOTO PASS

:ERROR
ECHO ^<^< errorlevel %ERRORLEVEL%


:PASS
