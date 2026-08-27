; Dafne Windows Installer Script
; Downloads Python, creates a virtual environment, and installs Dafne via pip.
; Mirrors the approach used in dafne-mac-pkg-build.sh for macOS.

#define MyAppName "Dafne"
#define MyAppVersion "2.1.0a0"
#define MyAppPublisher "Dafne-imaging"
#define MyAppURL "https://dafne.network/"
#define PythonVersion "3.13.13"
#define PythonDirName "Python313"
#define PipPackage "dafne"

[Setup]
AppId={{451322B2-10C5-4BA0-88DC-BB8933F78678}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=auto
DefaultGroupName={#MyAppName}
PrivilegesRequired=admin
OutputDir=..\dist
OutputBaseFilename=dafne_windows_setup_{#MyAppVersion}
SetupIconFile=..\icons\dafne_icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\icons\dafne_icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\venv\Scripts\dafne.exe"; IconFilename: "{app}\dafne_icon.ico"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\venv\Scripts\dafne.exe"; IconFilename: "{app}\dafne_icon.ico"; Tasks: desktopicon

[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"

[Code]
var
  NeedPythonInstall: Boolean;
  PythonExePath: String;
  WhlDirPage: TInputDirWizardPage;
  LocalWhlDir: String;

{ Find an existing Python 3.11 installation in common locations }
function GetSystemPythonExe: String;
var
  PF64Path, LocalPath, RootPath: String;
begin
  PF64Path  := ExpandConstant('{pf64}\{#PythonDirName}\python.exe');
  LocalPath := ExpandConstant('{localappdata}\Programs\Python\{#PythonDirName}\python.exe');
  RootPath  := ExpandConstant('\{#PythonDirName}\python.exe');

  if FileExists(PF64Path) then
    Result := PF64Path
  else if FileExists(LocalPath) then
    Result := LocalPath
  else if FileExists(RootPath) then
    Result := RootPath
  else
    Result := '';
end;

procedure InitializeWizard;
begin
  PythonExePath     := GetSystemPythonExe;
  NeedPythonInstall := (PythonExePath = '');
  if NeedPythonInstall then
    Log('Python {#PythonVersion} not found – will download installer.')
  else
    Log('Found existing Python at: ' + PythonExePath);

  { Optional page: let the user point pip at a local folder of .whl files }
  { (e.g. for offline installs or testing pre-built wheels) instead of,   }
  { or in addition to, downloading packages from PyPI.                   }
  WhlDirPage := CreateInputDirPage(wpSelectTasks,
    'Local Wheel Files (Optional)',
    'Specify a folder containing .whl files to use during installation',
    'If you have a local folder with pre-downloaded Python wheel (.whl) files for Dafne ' +
    'and/or its dependencies, select it below so pip can use it. Leave this blank to install ' +
    'normally from PyPI.',
    False, '');
  WhlDirPage.Add('');
  WhlDirPage.Values[0] := '';
end;

{ Allow the local-wheels folder field to be left blank; only validate it }
{ (must exist) when the user actually enters something.                 }
function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = WhlDirPage.ID then
  begin
    if (WhlDirPage.Values[0] <> '') and not DirExists(WhlDirPage.Values[0]) then
    begin
      MsgBox('The folder you specified does not exist:' + #13#10 + WhlDirPage.Values[0],
             mbError, MB_OK);
      Result := False;
    end;
  end;
end;

{ Download the Python installer before the file-copy phase so the user }
{ sees a clear "Preparing" step rather than an unexplained pause.       }
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
  Url, Dest, Cmd: String;
begin
  Result := '';

  if not NeedPythonInstall then
    Exit;

  Log('Downloading Python {#PythonVersion} installer...');
  Url  := 'https://www.python.org/ftp/python/{#PythonVersion}/python-{#PythonVersion}-amd64.exe';
  Dest := ExpandConstant('{tmp}') + '\python_installer.exe';
  Cmd  := '-Command "& { $ProgressPreference = ''SilentlyContinue''; ' +
          '[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; ' +
          'Invoke-WebRequest -Uri ''' + Url + ''' ' +
          '-OutFile ''' + Dest + ''' -UseBasicParsing }"';

  if not Exec('powershell.exe', Cmd, '', SW_SHOW, ewWaitUntilTerminated, ResultCode)
     or (ResultCode <> 0) then
  begin
    Result := 'Failed to download Python {#PythonVersion}.' + #13#10 +
              'Please check your internet connection and try again.';
  end;
end;

{ After Inno Setup has copied its own files, run the environment setup. }
procedure CurStepChanged(CurStep: TSetupStep);
var
  AppDir, VenvDir, VenvPython, VenvPip, FindLinksArg: String;
  ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then
    Exit;

  AppDir    := ExpandConstant('{app}');
  VenvDir   := AppDir + '\venv';
  VenvPython := VenvDir + '\Scripts\python.exe';
  VenvPip   := VenvDir + '\Scripts\pip.exe';

  { --- Local wheel folder, if the user provided one -------------------- }
  LocalWhlDir := WhlDirPage.Values[0];
  FindLinksArg := '';
  if LocalWhlDir <> '' then
  begin
    Log('Using local wheel folder: ' + LocalWhlDir);
    FindLinksArg := ' --find-links="' + LocalWhlDir + '"';
  end;

  { --- Install Python -------------------------------------------------- }
  if NeedPythonInstall then
  begin
    Log('Installing Python {#PythonVersion} silently...');
    if not Exec(ExpandConstant('{tmp}') + '\python_installer.exe',
                'InstallAllUsers=1 PrependPath=0 ' +
                'Include_test=0 Include_doc=0 Include_launcher=1',
                '', SW_SHOW, ewWaitUntilTerminated, ResultCode)
       or (ResultCode <> 0) then
    begin
      MsgBox('Failed to install Python {#PythonVersion}.' + #13#10 +
             'Please install it manually from https://www.python.org/ and re-run this installer.',
             mbError, MB_OK);
      Exit;
    end;

    PythonExePath := GetSystemPythonExe;
    if PythonExePath = '' then
    begin
      MsgBox('Python was installed but could not be located.' + #13#10 +
             'Please install Python {#PythonVersion} manually and re-run this installer.',
             mbError, MB_OK);
      Exit;
    end;
    Log('Python installed at: ' + PythonExePath);
  end;

  { --- Remove any existing virtual environment ------------------------- }
  if DirExists(VenvDir) then
  begin
    Log('Removing existing virtual environment...');
    DelTree(VenvDir, True, True, True);
  end;

  { --- Create virtual environment -------------------------------------- }
  Log('Creating virtual environment in: ' + VenvDir);
  if not Exec(PythonExePath, '-m venv "' + VenvDir + '"',
              '', SW_SHOW, ewWaitUntilTerminated, ResultCode)
     or (ResultCode <> 0) then
  begin
    MsgBox('Failed to create Python virtual environment.' + #13#10 +
           'Please contact support or check the log file.',
           mbError, MB_OK);
    Exit;
  end;
  
  { --- Grant Users modify access on the venv so pip works unprivileged -- }
  Log('Setting permissions on virtual environment...');
  Exec('icacls.exe', '"' + VenvDir + '" /grant Users:(OI)(CI)M /T /Q',
       '', SW_SHOW, ewWaitUntilTerminated, ResultCode);

  { --- Upgrade pip inside the venv ------------------------------------- }
  Log('Upgrading pip...');
  Exec(VenvPython, '-m pip install --upgrade pip',
       '', SW_SHOW, ewWaitUntilTerminated, ResultCode);

  { --- Install Dafne --------------------------------------------------- }
  Log('Installing {#PipPackage}=={#MyAppVersion}...');
  if not Exec(VenvPip, 'install --upgrade {#PipPackage}' + FindLinksArg,
              '', SW_SHOW, ewWaitUntilTerminated, ResultCode)
     or (ResultCode <> 0) then
  begin
    MsgBox('Failed to install Dafne.' + #13#10 +
           'Please check your internet connection and try again.',
           mbError, MB_OK);
    Exit;
  end;

  Log('Upgrading flexidep...');
  Exec(VenvPython, '-m pip install --upgrade flexidep' + FindLinksArg,
       '', SW_SHOW, ewWaitUntilTerminated, ResultCode);

  { --- Install pyradiomics ---------------------------------------------- }
  Log('Installing pyradiomics...');
  if not Exec(VenvPip, 'install --upgrade pyradiomics' + FindLinksArg,
              '', SW_SHOW, ewWaitUntilTerminated, ResultCode)
     or (ResultCode <> 0) then
  begin
    MsgBox('Failed to install pyradiomics.' + #13#10 +
           'Please check your internet connection and try again.',
           mbError, MB_OK);
    Exit;
  end;

  { --- Replace SimpleITK with SimpleITK-SimpleElastix ------------------- }
  Log('Removing SimpleITK...');
  Exec(VenvPip, 'uninstall -y SimpleITK',
       '', SW_SHOW, ewWaitUntilTerminated, ResultCode);

  Log('Installing SimpleITK-SimpleElastix...');
  if not Exec(VenvPip, 'install --upgrade SimpleITK-SimpleElastix' + FindLinksArg,
              '', SW_SHOW, ewWaitUntilTerminated, ResultCode)
     or (ResultCode <> 0) then
  begin
    MsgBox('Failed to install SimpleITK-SimpleElastix.' + #13#10 +
           'Please check your internet connection and try again.',
           mbError, MB_OK);
    Exit;
  end;

  Log('Dafne installation completed successfully.');
end;
