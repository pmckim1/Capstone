mkdir -p ArticleTexts
mkdir -p WhooshIndex
mkdir -p Cache
mkdir -p Output

# System viz tools
sudo yum install -y tmux git htop

# Some python modules requires gcc and g++
sudo yum install -y python3 python3-devel
sudo yum group install -y "Development Tools"

# python-igraph requires cmake3.16 or highter.
# Get the binary distribution.
wget -nc -P ~/ https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.tar.gz
# UNpack it.
tar -zxvf ~/cmake-3.20.0-linux-x86_64.tar.gz -C ~/
# Move the cmake tool into place
sudo cp ~/cmake-3.20.0-linux-x86_64/bin/cmake /usr/bin/cmake-3.20
# Symlink the cmake executable to use it.
sudo ln -s /usr/bin/cmake-3.20 /usr/bin/cmake
# python-igraph needs the usr share data.
sudo cp -r ~/cmake-3.20.0-linux-x86_64/share/cmake-3.20 /usr/share/

# Now install the real igraph.
sudo python3 -m pip install -r requirements.txt

