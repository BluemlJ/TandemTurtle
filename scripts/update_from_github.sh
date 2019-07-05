 
sudo chmod -R 777 ../chess
echo "----------Updating TandemTurtle"
cd ../
git pull

echo "----------Removing chess folder in TandemTurtle"
rm -r chess #new chess folder comes from python chess (see below)

echo "----------Updating Python Chess"
cd ../python-chess
git pull


echo "----------Updating server"
cd ../tinyChessServer
alr=`git pull | grep 'Already'` 
if test -z "$alr"
then
      echo "--------Installing server"
      npm install
      cd vue-frontend/
      npm install
      cd ..
else
      echo "Already up to date."
fi

echo "----------Removing chess folder in TandemTurtle"
rm -r backend/chess #new chess folder comes from python chess (see below)


echo "----------Copying chess folder"
cd ../python-chess
cp -r chess ../TandemTurtle
cp -r chess ../tinyChessServer/backend/

