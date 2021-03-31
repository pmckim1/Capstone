mkdir -p ArticleTexts
mkdir -p WhooshIndex
mkdir -p Cache
mkdir -p Output

sudo yum install python3 python3-devel
sudo yum group install "Development Tools"
sudo python3 -m pip install -r requirements.txt

