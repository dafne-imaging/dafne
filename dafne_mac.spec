# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['dafne.py'],
             pathex=['/Users/francesco/dafne/dafne'],
             binaries=[],
             datas=[('LICENSE', '.'), ('ui/images/*', 'ui/images/')],
             hiddenimports = ['pydicom', 
              'SimpleITK',
              'potrace.bezier',
              'potrace.agg',
              'potrace.agg.curves',
              'tensorflow',
              'keras',
              'skimage',
              'nibabel',
              'dl'],
             hookspath=['.'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='dafne',
          debug=False,
          icon='dafne_icon.ico',
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='dafne')
