// frontend/src/App.js

import React from 'react';
// 1. Change the import from ImageUploader to FileUpload
import FileUpload from './components/FileUpload'; 
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Welcome to the Bovine Analysis Tool</h1>
      </header>
      <main>
        {/* 2. Change the component name here */}
        <FileUpload /> 
      </main>
    </div>
  );
}

export default App;
