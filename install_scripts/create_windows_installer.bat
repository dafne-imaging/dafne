python -V
python update_version.py
pyinstaller dafne_win.spec --noconfirm
"C:\Program Files (x86)\Inno Setup 6\Compil32.exe" /cc dafne_win.iss