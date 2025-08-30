/* frontend/src/components/FileUpload.css */

.upload-container {
  max-width: 900px;
  margin: 2rem auto;
  padding: 2rem;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background-color: #ffffff;
  border-radius: 12px;
  box-shadow: 0 8px Gist;
  text-align: center;
}

.tabs {
  display: flex;
  border-bottom: 2px solid #e0e0e0;
  margin-bottom: 1.5rem;
}

.tab {
  padding: 1rem 1.5rem;
  cursor: pointer;
  font-size: 1.1rem;
  font-weight: 500;
  color: #666;
  border-bottom: 3px solid transparent;
  transition: all 0.3s ease;
}

.tab.active {
  color: #007bff;
  border-bottom-color: #007bff;
}

.file-input-area {
  border: 2px dashed #007bff;
  border-radius: 8px;
  padding: 2rem;
  margin-bottom: 1.5rem;
  background-color: #f8f9fa;
}

.file-input-area p {
  margin: 0;
  color: #555;
}

.file-input {
  width: 100%;
  padding: 1rem;
}

.analyze-button {
  padding: 0.8rem 2rem;
  font-size: 1.1rem;
  font-weight: bold;
  color: white;
  background-color: #007bff;
  border: none;
  border-radius: 50px;
  cursor: pointer;
  transition: background-color 0.3s ease, box-shadow 0.3s ease;
}

.analyze-button:hover {
  background-color: #0056b3;
  box-shadow: 0 4px 15px rgba(0, 123, 255, 0.4);
}

.analyze-button:disabled {
  background-color: #a0a0a0;
  cursor: not-allowed;
  box-shadow: none;
}

.error-message {
  color: #dc3545;
  margin-top: 1rem;
  font-weight: 500;
}

.results-section {
  margin-top: 2rem;
  display: flex;
  flex-wrap: wrap;
  gap: 2rem;
  align-items: flex-start;
  justify-content: center;
  text-align: left;
}

.media-preview, .results-card {
  flex: 1;
  min-width: 300px;
  max-width: 400px;
}

.results-card {
  background: #f9f9f9;
  padding: 1.5rem;
  border-radius: 8px;
  border-left: 5px solid #007bff;
}

.results-card h3, .results-card h4 {
  margin-top: 0;
  color: #333;
}

.results-card hr {
  border: none;
  border-top: 1px solid #e0e0e0;
  margin: 1rem 0;
}

.results-card p {
  margin: 0.5rem 0;
  color: #444;
}

.preview-image {
  max-width: 100%;
  border-radius: 8px;
}
