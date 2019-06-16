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
  * Make sure the bughouse chess variant of python-chess can be found.
You can download the python-chess version here:
https://github.com/TimSchneider42/python-chess.
To install, run `python3 setup.py build` and `python3 setup.py install`
  * Make sure you have nodejs installed [https://nodejs.org/]
  * Make sure you have all the necessary python packages installed (do `pip install -r requirements.txt`)
  * Make sure you have netcat installed [http://netcat.sourceforge.net/]
  * Use the tinyChessServer: [https://github.com/MoritzWillig/tinyChessServer/]
  * In the tinyChessServer folder run `npm install`
  * In the tinyChessServer/vue-frontend folder run `npm install`

# How to contribute (more installs!)
  * make sure `autopep8` is installed for python3
  * copy the file `pre-commit` to `.git/hooks/pre-commit`
  * make the file executable

# How to run

To run on the server (and see the match visualized 'in action'), please:
  * Run the server using `node index.js` in the server folder. You might need sudo.
  * Run the frontend by running `npm run serve` in the `vue-frontend` folder
  * Open a webbrowser on `localhost:8080`
  * Start our program by running `python3 main.py`
  * Type in `go` in the `node index.js`-console or click on `go` Button in the frontend
  * Watch a beautiful game of Bughouse chess in action!

If the server is not runinng, it's autocatically running in CLI mode (like sunsetter or other chess engines)

# Contributors
Jannis Blüml, Maximilian Otte, Florian Netzer, Magdalena Wache, Lars Wolf

# License
GPL v3.0
