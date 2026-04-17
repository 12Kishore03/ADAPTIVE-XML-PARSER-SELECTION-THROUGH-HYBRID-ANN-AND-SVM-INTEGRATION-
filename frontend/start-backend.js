const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log('🚀 Starting Backend Server...');

// Path to backend
const backendPath = path.join(__dirname, '..', 'backend', 'test_app.py');

// Check if backend file exists
if (!fs.existsSync(backendPath)) {
    console.error('❌ Backend file not found:', backendPath);
    console.log('💡 Please make sure your backend files are in the backend/ folder');
    process.exit(1);
}

console.log('✅ Found backend file:', backendPath);

// Start the Python backend
console.log('🐍 Starting Python backend server...');
const backendProcess = spawn('python', [backendPath], {
    cwd: path.join(__dirname, '..', 'backend'),
    stdio: 'inherit'
});

backendProcess.on('error', (error) => {
    console.error('❌ Failed to start backend:', error.message);
    console.log('💡 Make sure Python is installed and in your PATH');
    console.log('💡 You can install Python from: https://python.org');
});

backendProcess.on('close', (code) => {
    if (code !== 0) {
        console.log(`❌ Backend process exited with code ${code}`);
        console.log('💡 Trying to start React frontend anyway...');
    }
});

// Handle app termination
process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down backend...');
    backendProcess.kill();
    process.exit();
});

process.on('SIGTERM', () => {
    console.log('\n🛑 Shutting down backend...');
    backendProcess.kill();
    process.exit();
});

// Give backend time to start
setTimeout(() => {
    console.log('✅ Backend should be running on http://localhost:5000');
    console.log('🌐 Frontend will start on http://localhost:3000');
}, 2000);