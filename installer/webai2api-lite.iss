; webai2api Lite 版安装包脚本（Inno Setup 6+）
; 用法：先构建 Lite 版到 dist-lite\webai2api，然后用 ISCC.exe 编译本文件。
;   & "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\webai2api-lite.iss

#define MyAppName "webai2api 客户版"
#define MyAppVersion "0.1.26"
#define MyAppPublisher "DuRuoxian"
#define MyAppURL "https://github.com/duruoxian/webai2api"
#define MyAppExeName "webai2api.exe"
#define SourceDir "..\dist-lite\webai2api"

[Setup]
AppId={{7C1E2B54-9A3D-4F6B-8E21-5D0A9C3B7F12}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\webai2api-lite
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist-installer
OutputBaseFilename=webai2api-lite-setup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupIconFile=..\assets\webai2api-lite.ico
CloseApplications=yes

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"; Flags: checkedonce
Name: "autostart"; Description: "开机自动启动（托盘后台运行）"; GroupDescription: "附加任务:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
; 不打包 data 目录：保留用户数据库与授权

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "webai2api-lite"; ValueData: """{app}\{#MyAppExeName}"""; Flags: uninsdeletevalue; Tasks: autostart

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "立即启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 卸载时保留 data（用户授权/账号/用量），仅清理日志与端口文件
Type: files; Name: "{app}\webai2api.log"
Type: files; Name: "{app}\current_port.txt"
