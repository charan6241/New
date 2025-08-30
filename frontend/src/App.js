// frontend/src/App.js

import React from 'react';
import ImageUploader from './components/ImageUploader';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Welcome to the Bovine Analysis Tool</h1>
      </header>
      <main>
        <ImageUploader />
      </main>
    </div>
  );
}

export default App;
