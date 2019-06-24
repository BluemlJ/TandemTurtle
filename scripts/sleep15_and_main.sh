# sub script for run_game.sh

cd ../TandemTurtle/
sleep 7
python3 main.py

echo "Press any key to continue"
echo "waiting for the keypress"
while [ true ] ; do
read -t 3 -n 1
if [ $? = 0 ] ; then
exit ;
else
fi
done
