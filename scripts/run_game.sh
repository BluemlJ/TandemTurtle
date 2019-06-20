# run the server, then run the main function. Saves some effort to type in every command.

chmod 755 sleep5_and_run_serve.sh
chmod 755 sleep15_and_main.sh
cd ../
cd ../tinyChessServer/
sudo xterm -e sudo node index.js & xterm -e ../TandemTurtle/scripts/sleep5_and_run_serve.sh & xterm -e ../TandemTurtle/scripts/sleep15_and_main.sh
