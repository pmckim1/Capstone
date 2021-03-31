mkdir -p ArticleTexts
mkdir -p WhooshIndex
mkdir -p Cache
mkdir -p Output

sudo yum install -y tmux git
sudo yum install -y python3 python3-devel
sudo yum group install -y "Development Tools"
sudo python3 -m pip install -r requirements.txt

