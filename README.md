# How to install
Make sure the bughouse chess variant of python-chess can be found. You can download the python-chess version here: https://github.com/TimSchneider42/python-chess. To install, run `python3 setup.py build` and `python3 setup.py install`

# How to run
`python3 main.py` runs the main programm - either locally or on the server.

To run on the server (and see the match visualized 'in action'), please:

  * Use tinyChessServer https://github.com/MoritzWillig/tinyChessServer/projects
  * run the server using `node index.js`
  * run the frontend by changing to `vue-frontend` folder and running `npm run serve`
  * open a webbrowser on `localhost:8080`
  * start our program by running `python3 main.py`
  * type in `go` in the server console or click on `go` in the frontend
  * watch a beautiful game of bughouse chess in action!


# Contributors
Jannis Blüml, Maximilian Otte, Florian Netzer, Magdalena Wache, Lars Wolf

# License
GPL v3.0