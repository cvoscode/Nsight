@echo off

type ASCIIStartLogo.txt


call env\Scripts\activate

start http://127.0.0.1:8080/

echo If the app does not load automatically please refresh

waitress-serve app:app.server



