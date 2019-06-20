 

echo "----------Updating TandemTurtle"
cd ../
git pull

echo "----------Removing chess folder"
rm -r chess #new chess folder comes from python chess (see below)

echo "----------Updating Python Chess"
cd ../python-chess
git pull
echo "----------Copying chess folder"
cp -r chess ../TandemTurtle

echo "----------Updating server"
cd ../tinyChessServer
alr=`git pull | grep 'Already'` 
if test -z "$alr"
then
      echo "--------Installing server"
      npm install
      cd vue-frontend/
      npm install
else
      echo "Already up to date."
fi


