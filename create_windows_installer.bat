git pull
cd dl
git checkout master
git pull
cd ..
python update_version.py
pyinstaller dafne_win.spec --noconfirm
"C:\Program Files (x86)\Inno Setup 6\Compil32.exe" /cc dafne_win.iss