// frontend/src/components/FileUpload.js

import React, { useState } from 'react';
import axios from 'axios';
import './FileUpload.css'; // Make sure you have the corresponding CSS file

const FileUpload = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [predictionData, setPredictionData] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('image'); // 'image' or 'video'

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
            setPreview(URL.createObjectURL(file));
            setPredictionData(null);
            setError(null);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError("Please select a file first.");
            return;
        }

        setIsLoading(true);
        setError(null);
        setPredictionData(null);

        const formData = new FormData();
        formData.append("file", selectedFile);
        
        const endpoint = activeTab === 'image' ? '/predict-image' : '/predict-video';
        
        // ==============================================================================
        // IMPORTANT: THIS IS THE URL YOU MUST CHANGE FOR DEPLOYMENT
        // ==============================================================================
        // For local testing, use: "http://127.0.0.1:8000"
        // For the final live website, use your public Render URL, for example:
        // const BASE_URL = "https://cattle-vision-api.onrender.com";
        // ==============================================================================
        const BASE_URL = "https://new-idsi.onrender.com"; // <-- PASTE YOUR RENDER URL HERE
        
        const API_URL = BASE_URL + endpoint;

        try {
            const response = await axios.post(API_URL, formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            setPredictionData(response.data);
        } catch (err) {
            console.error("Error uploading file:", err);
            setError("Analysis failed. Please ensure the backend server is running and accessible.");
        } finally {
            setIsLoading(false);
        }
    };
    
    // Helper function to render results consistently for both image and video
    const renderResults = () => {
        const data = predictionData.prediction || predictionData.summary;
        if (!data) return null;

        return (
            <>
                <p><strong>Animal Type:</strong> {data.animal_type}</p>
                <p><strong>Predicted Breed:</strong> {data.breed}</p>
                <p><strong>Visual Health Status:</strong> {data.health_status}</p>
                <hr />
                <h4>Breed Information</h4>
                <p><strong>Milk Yield:</strong> {predictionData.breed_info.milk_yield || 'N/A'}</p>
                <p><strong>Weight Range:</strong> {predictionData.breed_info.weight_range || 'N/A'}</p>
                <p><i>{predictionData.breed_info.info}</i></p>
            </>
        );
    };

    return (
        <div className="upload-container">
            <div className="tabs">
                <div className={`tab ${activeTab === 'image' ? 'active' : ''}`} onClick={() => setActiveTab('image')}>
                    🖼️ Image Analysis
                </div>
                <div className={`tab ${activeTab === 'video' ? 'active' : ''}`} onClick={() => setActiveTab('video')}>
                    🎬 Video Analysis
                </div>
            </div>

            <div className="file-input-area">
                <p>Select an {activeTab} to analyze</p>
                <input 
                    type="file" 
                    key={activeTab} 
                    accept={activeTab === 'image' ? "image/*" : "video/*"} 
                    onChange={handleFileChange} 
                    className="file-input" 
                />
            </div>
            
            <button onClick={handleUpload} disabled={isLoading || !selectedFile} className="analyze-button">
                {isLoading ? 'Analyzing...' : `Analyze ${activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}`}
            </button>

            {error && <p className="error-message">{error}</p>}

            <div className="results-section">
                {preview && (
                    <div className="media-preview">
                        <h3>Preview</h3>
                        {activeTab === 'image' ? (
                            <img src={preview} alt="Selected Preview" className="preview-image" />
                        ) : (
                            <video src={preview} controls width="100%" />
                        )}
                    </div>
                )}

                {isLoading && <p>Loading results...</p>}
                {predictionData && (
                    <div className="results-card">
                        <h3>Analysis Results</h3>
                        {renderResults()}
                    </div>
                )}
            </div>
        </div>
    );
};

export default FileUpload;

