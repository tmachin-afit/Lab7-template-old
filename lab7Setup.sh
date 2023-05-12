## RUNNING ON ACE
# ensure updated system
apt update && apt upgrade -y
snap refresh
snap install --classic code
# install pip (python package manager) if not installed already
apt install -y python3-tk
python3 -m pip install --upgrade pip
# install python requirements
python3 -m pip install -r requirements.txt
mkdir /opt/data/Oxford_Inertial_Tracking_Dataset
# copy data from local to new directory
cp -a ../data/Oxford_Inertial_Tracking_Dataset/. /opt/data/Oxford_Inertial_Tracking_Dataset/

## RUNNING ON VM/METAL
# # ensure updated system
# sudo apt update && sudo apt upgrade -y
# sudo snap refresh
# sudo snap install --classic code
# # install pip (python package manager) if not installed already
# sudo apt install -y python3-pip
# python3 -m pip install --upgrade pip
# # install python requirements
# python3 -m pip install -r requirements.txt
# sudo mkdir /opt/data
# sudo cp -a ../data/. /opt/data/