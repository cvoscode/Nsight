@echo off

type ASCIIStartLogo.txt


call env\Scripts\activate
echo App is starting...
start http://127.0.0.1:8080/
echo If the app does not load automatically please refresh in a few seconds

waitress-serve app:app.server




