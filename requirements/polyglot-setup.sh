# install dev tools
apt-get update && apt-get install -y apt-transport-https
apt-get install libicu-dev
apt-get install python3-dev

#install things in this order
pip install pycld2
pip install polyglot
pip install --no-binary=:pyicu: pyicu

# all other packages
pip install -r requirements-polyglot.txt

# danish-specific downloads for polyglot
polyglot download pos2.da
polyglot download embeddings2.da
polyglot download ner2.da