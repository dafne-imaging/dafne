# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a_validate = Analysis(['batch_validate.py'],
             pathex=['C:\\dafne\\dafne'],
             binaries=[('libpotrace-0.dll', '.')],
             datas=[('LICENSE', '.'), ('ui\\images\\*', 'ui\\images')],
			 hiddenimports = ['pydicom',
				'tensorflow',
				'potrace.bezier',
				'potrace.agg',
				'potrace.agg.curves',
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

pyz_validate = PYZ(a_validate.pure, a_validate.zipped_data,
             cipher=block_cipher)
exe_validate = EXE(pyz_validate,
          a_validate.scripts,
          [],
          exclude_binaries=True,
          name='dafne_validator',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          icon='dafne_validate.ico',
          console=True )
coll_validate = COLLECT(exe_validate,
               a_validate.binaries,
               a_validate.zipfiles,
               a_validate.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='dafne_validate')


			   
