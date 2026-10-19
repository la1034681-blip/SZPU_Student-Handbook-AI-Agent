@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   深圳职业技术大学 学生手册 智能问答系统
echo ============================================
echo.
echo   正在启动服务，请稍候...
echo.
echo   进入页面需要输入访问口令（见项目根 .env 的 ACCESS_CODE）
echo   关闭本窗口即停止服务
echo.
echo ============================================
echo.

set PY=C:\Users\lijiahui\miniconda3\envs\agent\python.exe
if not exist "%PY%" (
    echo [错误] 找不到 Python 环境：
    echo   %PY%
    echo 请把本窗口截图发给助手。
    pause
    exit /b 1
)

if not exist "app.py" (
    echo [错误] 当前目录下找不到 app.py
    echo 当前目录：%cd%
    echo 请把本窗口截图发给助手。
    pause
    exit /b 1
)

rem 自动找一个空闲端口：本机还装了别的 Streamlit 项目（也爱用 8501），
rem 硬写端口会撞车 —— 谁先启动谁占住，后启动的直接报错退出，
rem 而浏览器已经开了 8501，你看到的就是别人的页面 / 旧进程的残页。
for /f "delims=" %%p in ('"%PY%" _pick_port.py') do set PORT=%%p
if "%PORT%"=="" (
    echo [错误] 没能找到可用端口（8501-8520 都被占用）。
    echo 请先关掉别的网页服务窗口，再双击本文件。
    pause
    exit /b 1
)

echo   服务地址：http://localhost:%PORT%
echo   约 10 秒后会自动打开浏览器；
echo   若没自动打开，请手动复制上面这个地址访问。
echo.

start "" /b cmd /c "ping -n 11 127.0.0.1 >nul & start http://localhost:%PORT%"

"%PY%" -m streamlit run app.py --server.headless true --server.port %PORT%

echo.
echo [服务已结束] 若上方有报错信息，请截图发给助手。
pause
