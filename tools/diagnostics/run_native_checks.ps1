param([string]$Python = "$PSScriptRoot\..\..\venv\python.exe")
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath("$PSScriptRoot\..\..")
$pythonPath = [IO.Path]::GetFullPath($Python)
$cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0' }
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$install = & $vswhere -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $install) { throw 'Visual Studio 2022 C++ tools required.' }
$obj = 'build\temp.win-amd64-cpython-312\Release'
if (-not (Test-Path -LiteralPath "$root\$obj\src\vonnx_engine.obj")) { throw 'Run build.ps1 first.' }
$batch = "$root\build\native_checks.cmd"
$vcvars = "$install\VC\Auxiliary\Build\vcvars64.bat"
$lines = @('@echo off','chcp 65001 >nul',"call `"$vcvars`"",'if errorlevel 1 exit /b %errorlevel%',"cd /d `"$root`"",
    "`"$pythonPath`" -c `"import sys;sys.path.insert(0,'tests');from pathlib import Path;from test_native import fixture;fixture(Path('build/specialization_fixture.onnx'))`"",'if errorlevel 1 exit /b %errorlevel%',
    "cl /nologo /std:c++17 /EHsc /O2 /utf-8 /Iinclude /I`"$cuda\include`" tests\test_scheduler.cpp src\continuous_scheduler.cpp /Febuild\test_scheduler.exe /Fobuild\",'if errorlevel 1 exit /b %errorlevel%',
    'build\test_scheduler.exe','if errorlevel 1 exit /b %errorlevel%',
    "cl /nologo /std:c++17 /EHsc /O2 /utf-8 /MD /Iinclude /Iinclude\onnxruntime /I`"$cuda\include`" tests\test_engine_isolation.cpp $obj\src\continuous_scheduler.obj $obj\src\onnx_runner.obj $obj\src\vonnx_engine.obj $obj\paged_tensor_pool.cu.obj $obj\cuda_preprocess.cu.obj $obj\cuda_nms.cu.obj /Fevonnx_standalone_sdk\test_engine_isolation.exe /Fobuild\ /link /LTCG /LIBPATH:`"$cuda\lib\x64`" /LIBPATH:$obj cudart.lib onnxruntime.lib",'if errorlevel 1 exit /b %errorlevel%',
    "vonnx_standalone_sdk\test_engine_isolation.exe build\specialization_fixture.onnx `"$pythonPath`" `"$root\vonnx\precision.py`" `"$root\runs\software_wait\fixture_cache`"")
[IO.File]::WriteAllLines($batch,$lines,[Text.UTF8Encoding]::new($false))
$savedPath=$env:PATH
Push-Location $root
try {
    $env:PATH="$cuda\bin\x64;$root\venv\Lib\site-packages\torch\lib;$root\venv\Library\bin;$savedPath"
    & cmd /c $batch
    if ($LASTEXITCODE -ne 0) { throw "Native checks failed ($LASTEXITCODE)." }
} finally { $env:PATH=$savedPath; Pop-Location }
