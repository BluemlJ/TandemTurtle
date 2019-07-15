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

# How to set up
  * Make sure the bughouse chess variant of [python-chess](https://github.com/TimSchneider42/python-chess.) can be found.
  * Make sure you have [nodejs](https://nodejs.org/) installed.
  * You may need to install [netcat](http://netcat.sourceforge.net/).
  * Use the [tinyChessServer](https://github.com/MoritzWillig/tinyChessServer/).
  * Run the TandemTurtle/scripts/update_from_github.sh script in order to install or reinstall and update all that is needed.
  * Alternative: If you have problems with the script, you can do it directly:
     * In the tinyChessServer folder run `npm install`.
     * In the tinyChessServer/vue-frontend folder also run `npm install`.
     * Copy the python-chess/chess folder into TandemTurtle/.
     * Also copy the python-chess/chess folder into tinyChessServer/backend.
     * Make sure you have all the necessary python packages installed (do `pip install -r requirements.txt` in the TandemTurtle folder).
  * In TandemTurtle create a folder `run/models` then put your model in this folder. In config.py, adjust INITIAL_MODEL_PATH to your models name. **TODO**: upload our model.
  * In tinyChessServer put the necessary config files (**TODO:** Upload our config.json files)

# How to run

To run on the server (and see the match visualized 'in action'), please:
  * Decide how many Agent threads you want and in config.py set GAME_AGENT_THREADS accordingly.

4 player mode: (playing against itself)
  * if you want do run the server separately: Run the server using `node index.js` in the server folder. You might need sudo.
  * otherwise, in config.py set SERVER_AUTOSTART to 1. (If you run the server separately, leave it at 0 !)
  * Start our program by running `python3 main.py`
  * As soon as you see `run status preparing: 4 of 4 ready clients` , open a webbrowser on `localhost:8081`.
  * Watch a beautiful game of Bughouse chess in action.

2/1/0 player mode (**TO DO:** Describe in README!)

If you want to run an older version you will have to do the following steps:
  * Run the server using `node index.js` in the server folder. You might need sudo.
  * Run the frontend by running `npm run serve` in the `vue-frontend` folder
  * Open a webbrowser on `localhost:8080`
  * Type in `go` in the `node index.js`-console or click on `go` Button in the frontend.
  
# Troubleshooting
  * If you suspect a problem with MCTS, you can play only with the neural Network by setting RUN_ON_NN_ONLY to `True` in config.py.

# How to contribute (more installs!)
  * make sure `autopep8` is installed for python3
  * copy the file `scripts/pre-commit` to `.git/hooks/pre-commit`
  * make the file executable

# Contributors
Jannis Blüml, Maximilian Otte, Florian Netzer, Magdalena Wache, Lars Wolf

# License
GPL v3.0
