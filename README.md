~~~~
                        __   __
           .,-;-;-,.   /'_\ /_'\ .,-;-;-,.
          _/_/_/_|_\_\) /    \ (/_/_|_\_\_\_
       '-<_><_><_><_>=/\     /\=<_><_><_><_>-'
         `/_/====/_/-'\_\   /_/'-\_\====\_\'

  _____             _             _____         _   _     
 |_   _|_ _ _ _  __| |___ _ __   |_   _|  _ _ _| |_| |___
   | |/ _` | ' \/ _` / -_) '  \    | || || | '_|  _| / -_)
   |_|\__,_|_||_\__,_\___|_|_|_|   |_| \_,_|_|  \__|_\___|


~~~~
Art by Joan Stark

# How to install
  * in TandemTurtle create a folder `run/models` then put your model in this folder. TODO: upload our model.
  * Make sure the bughouse chess variant of python-chess can be found.
You can download the python-chess version here:
https://github.com/TimSchneider42/python-chess.
  * Make sure you have nodejs installed [https://nodejs.org/]
  * Make sure you have all the necessary python packages installed (do `pip install -r requirements.txt`)
  * You may need to install netcat [http://netcat.sourceforge.net/]
  * Use the tinyChessServer: [https://github.com/MoritzWillig/tinyChessServer/]
  * Run the TandemTurtle/scripts/update_from_github.sh script in order to update and (if necessary) reinstall all that is needed.
  * If you have problems with the script, you can do it directly:
     * To install python-chess, run `python3 setup.py build` and `python3 setup.py install` in your python chess folder.
     * In the tinyChessServer folder run `npm install`
     * In the tinyChessServer/vue-frontend folder run `npm install`

# How to contribute (more installs!)
  * make sure `autopep8` is installed for python3
  * copy the file `pre-commit` to `.git/hooks/pre-commit`
  * make the file executable

# How to run

To run on the server (and see the match visualized 'in action'), please:
  * Start our program by running `python3 main.py`
  * As soon as you see `run status preparing: 4 of 4 ready clients` , open a webbrowser on `localhost:8080`.
  * Watch a beautiful game of Bughouse chess in action

If you want to run an older version you will have to do the following steps:
  * Run the server using `node index.js` in the server folder. You might need sudo.
  * Run the frontend by running `npm run serve` in the `vue-frontend` folder
  * Open a webbrowser on `localhost:8080`  
  * Type in `go` in the `node index.js`-console or click on `go` Button in the frontend.

# Contributors
Jannis Blüml, Maximilian Otte, Florian Netzer, Magdalena Wache, Lars Wolf

# License
GPL v3.0
