; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "Dafne"
#define MyAppVersion "1.1-alpha"
#define MyAppPublisher "Dafne-imaging"
#define MyAppURL "https://www.dafne.network/"
#define MyAppExeName "dafne.exe"
#define CalcTransformsName "calc_transforms.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{451322B2-10C5-4BA0-88DC-BB8933F78678}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=auto
DefaultGroupName={#MyAppName}
LicenseFile=C:\dafne\dafne\dist\dafne\LICENSE
; Uncomment the following line to run in non administrative install mode (install for current user only.)
;PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=C:\dafne
OutputBaseFilename=dafne_setup
SetupIconFile=C:\dafne\dafne\dafne_icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "C:\dafne\dafne\dist\dafne\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\dafne\dafne\dist\calc_transforms\{#CalcTransformsName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\dafne\dafne\dist\calc_transforms\{#CalcTransformsName}.manifest"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\dafne\dafne\dist\dafne\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{group}\Offline Transform Calculator"; Filename: "{app}\{#CalcTransformsName}"
Name: "{autodesktop}\Offline Transform Calculator"; Filename: "{app}\{#CalcTransformsName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

