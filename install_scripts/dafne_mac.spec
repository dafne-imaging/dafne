# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

a_dafne = Analysis(['../run_dafne.py'],
             pathex=['../src'],
             binaries=[],
             datas=[('../LICENSE', '.'), ('../src/dafne/resources/*', 'resources/')],
             hiddenimports = ['dafne',
              'pydicom',
              'SimpleITK',
              'tensorflow',
              'skimage',
              'nibabel',
              'dafne_dl',
              'cmath',
              'ormir-pyvoxel',
              'pyvistaqt',
              'pyvista',
              'vtk',
              'torch',
              'torchvision'],
             hookspath=['../pyinstaller_hooks'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

a_calc_tra = Analysis(['../calc_transforms.py'],
             pathex=['../src'],
             binaries=[],
             datas=[],
             hiddenimports=[
              'dafne',
              'pydicom', 
              'SimpleITK',
              'ormir-pyvoxel'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False) 

MERGE( (a_dafne, 'dafne', 'dafne'), (a_calc_tra, 'calc_transforms', 'calc_transforms') )

pyz_dafne = PYZ(a_dafne.pure, a_dafne.zipped_data,
             cipher=block_cipher)
exe_dafne = EXE(pyz_dafne,
          a_dafne.scripts,
          [],
          exclude_binaries=True,
          name='dafne',
          debug=False,
          icon='../icons/dafne_icon.icns',
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll_dafne = COLLECT(exe_dafne,
               a_dafne.binaries,
               a_dafne.zipfiles,
               a_dafne.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='dafne')

pyz_calc_tra = PYZ(a_calc_tra.pure, a_calc_tra.zipped_data,
             cipher=block_cipher)
exe_calc_tra = EXE(pyz_calc_tra,
          a_calc_tra.scripts,
          [],
          exclude_binaries=True,
          name='calc_transforms',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          icon='../icons/calctransform_ico.ico',
          console=True )
coll_calc_tra = COLLECT(exe_calc_tra,
               a_calc_tra.binaries,
               a_calc_tra.zipfiles,
               a_calc_tra.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='calc_transforms')

app = BUNDLE(coll_dafne,
    name='Dafne.app',
    icon='../icons/dafne_icon.icns',
    bundle_identifier='network.dafne.dafne',
    version='1.8-alpha3')
