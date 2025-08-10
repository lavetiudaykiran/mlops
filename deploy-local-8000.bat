@echo off
echo ========================================
echo   MLOps Housing API - Local Deployment
echo         (Running on Port 8000)
echo ========================================

echo.
echo [1/6] Pulling latest image from Docker Hub...
docker pull abhansi/mlops-housing-api:latest
if %errorlevel% neq 0 (
    echo ERROR: Failed to pull image from Docker Hub!
    echo Make sure you have internet connection and Docker is running.
    pause
    exit /b 1
)

echo.
echo [2/6] Stopping existing containers...
docker stop mlops-housing-api 2>nul
docker stop housing-api 2>nul  
docker stop test-api-8000 2>nul
echo Containers stopped.

echo.
echo [3/6] Removing existing containers...
docker rm mlops-housing-api 2>nul
docker rm housing-api 2>nul
docker rm test-api-8000 2>nul
echo Containers removed.

echo.
echo [4/6] Creating required directories...
if not exist "logs" mkdir logs
echo Logs directory created: %cd%\logs

echo.
echo [5/6] Starting MLOps Housing API container...
docker run -d ^
    --name mlops-housing-api ^
    -p 8000:5000 ^
    -v %cd%/logs:/app/logs ^
    -e PYTHONPATH=/app ^
    --restart unless-stopped ^
    abhansi/mlops-housing-api:latest

if %errorlevel% neq 0 (
    echo ERROR: Container failed to start!
    echo Checking container logs...
    docker logs mlops-housing-api 2>nul
    pause
    exit /b 1
)

echo Container started successfully!

echo.
echo [6/6] Testing API connectivity...
echo Waiting for API to initialize...
timeout /t 8 /nobreak >nul

curl -s http://localhost:8000/metrics >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ API is responding successfully!
) else (
    echo ⚠ API not responding yet. It may need more time to start.
    echo Check logs with: docker logs mlops-housing-api
)

echo.
echo ========================================
echo        DEPLOYMENT COMPLETED!
echo ========================================
echo.
echo 🚀 MLOps Housing API is running at:
echo    http://localhost:8000
echo.
echo 📊 Available Endpoints:
echo    • Metrics:  http://localhost:8000/metrics
echo    • Predict:  http://localhost:8000/predict  
echo    • Retrain:  http://localhost:8000/retrain
echo.
echo 🧪 Test Commands:
echo    curl http://localhost:8000/metrics
echo.
echo    curl -X POST http://localhost:8000/predict ^
echo      -H "Content-Type: application/json" ^
echo      -d "{\"features\": [8.3252, 41.0, 6.984, 1.024, 322.0, 2.555, 37.88, -122.23]}"
echo.
echo 📝 Management Commands:
echo    • View logs:     docker logs mlops-housing-api
echo    • Stop API:      docker stop mlops-housing-api
echo    • Remove container: docker rm mlops-housing-api
echo    • Container status: docker ps
echo.
echo 📁 Logs location: %cd%\logs
echo.

echo Container status:
docker ps | findstr mlops-housing-api

echo.
echo Press any key to exit...
pause >nul