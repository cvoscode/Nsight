@echo off

type ASCIIStartLogo.txt


call rvizenv\Scripts\activate

start http://127.0.0.1:8080/

echo If the app loads automatically please refresh

waitress-serve app:app.server



