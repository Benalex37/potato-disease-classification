@echo off
REM Starts the React dev server on http://localhost:3000
REM react-scripts 4 uses webpack 4, whose MD4 hashing needs the legacy
REM OpenSSL provider on Node 17+. Without this it dies with ERR_OSSL_EVP_UNSUPPORTED.
set NODE_OPTIONS=--openssl-legacy-provider
cd /d "%~dp0frontend"
call npm start
