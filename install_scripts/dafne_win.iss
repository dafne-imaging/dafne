; Dafne Windows Installer Script
; Downloads Python, creates a virtual environment, and installs Dafne via pip.
; Mirrors the approach used in dafne-mac-pkg-build.sh for macOS.

#define MyAppName "Dafne"
#define MyAppVersion "2.0.1b0"
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

[Dirs]
Name: "{app}\venv"; Permissions: users-modify

[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"

[Code]
var
  NeedPythonInstall: Boolean;
  PythonExePath: String;

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
  AppDir, VenvDir, VenvPython, VenvPip: String;
  ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then
    Exit;

  AppDir    := ExpandConstant('{app}');
  VenvDir   := AppDir + '\venv';
  VenvPython := VenvDir + '\Scripts\python.exe';
  VenvPip   := VenvDir + '\Scripts\pip.exe';

  { --- Install Python -------------------------------------------------- }
  if NeedPythonInstall then
  begin
    Log('Installing Python {#PythonVersion} silently...');
    if not Exec(ExpandConstant('{tmp}') + '\python_installer.exe',
                '/quiet InstallAllUsers=1 PrependPath=0 ' +
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

  { --- Upgrade pip inside the venv ------------------------------------- }
  Log('Upgrading pip...');
  Exec(VenvPython, '-m pip install --upgrade pip',
       '', SW_SHOW, ewWaitUntilTerminated, ResultCode);

  { --- Install Dafne --------------------------------------------------- }
  Log('Installing {#PipPackage}=={#MyAppVersion}...');
  if not Exec(VenvPip, 'install {#PipPackage}=={#MyAppVersion}',
              '', SW_SHOW, ewWaitUntilTerminated, ResultCode)
     or (ResultCode <> 0) then
  begin
    MsgBox('Failed to install Dafne.' + #13#10 +
           'Please check your internet connection and try again.',
           mbError, MB_OK);
    Exit;
  end;

  Log('Dafne installation completed successfully.');
end;
