# How to install
  * Make sure the bughouse chess variant of python-chess can be found. 
You can download the python-chess version here: 
https://github.com/TimSchneider42/python-chess. 
To install, run `python3 setup.py build` and `python3 setup.py install`
  * Make sure you have nodejs installed [https://nodejs.org/]
  * Make sure you have websocket-client installed (eg. do `pip install websocket-client`)
  * Make sure you have netcat installed [http://netcat.sourceforge.net/]
  * Use the tinyChessServer: [https://github.com/MoritzWillig/tinyChessServer/]
  * In the server folder run `npm install`
  * In the vue-frontend folder run `npm install`
# How to run

To run on the server (and see the match visualized 'in action'), please:


  * Run the server using `node index.js` in the server folder.
  * Run the frontend by running `npm run serve` in the `vue-frontend` folder
  * Open a webbrowser on `localhost:8080`
  * Start our program by running `python3 main.py`
  * Type in `go` in the server console or click on `go` in the frontend
  * Watch a beautiful game of Bughouse chess in action!


# Contributors
Jannis Blüml, Maximilian Otte, Florian Netzer, Magdalena Wache, Lars Wolf

# License
GPL v3.0