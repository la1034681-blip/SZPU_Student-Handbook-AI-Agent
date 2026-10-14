@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   深圳职业技术大学 学生手册 智能问答系统
echo ============================================
echo.
echo   正在启动服务，请稍候...
echo   约 10 秒后会自动打开浏览器
echo   若没有自动打开，请手动访问：http://localhost:8501
echo.
echo   进入页面需要输入访问口令（默认 SZPU666，
echo   可在项目根目录 .env 的 ACCESS_CODE 这一行修改）
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

start "" /b cmd /c "ping -n 11 127.0.0.1 >nul & start http://localhost:8501"

"%PY%" -m streamlit run app.py --server.headless true --server.port 8501

echo.
echo [服务已结束] 若上方有报错信息，请截图发给助手。
pause
