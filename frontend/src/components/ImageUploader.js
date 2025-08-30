// frontend/src/components/ImageUploader.js

import React, { useState } from 'react';
import axios from 'axios';

const ImageUploader = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewImage, setPreviewImage] = useState(null);
    const [predictionData, setPredictionData] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
            setPreviewImage(URL.createObjectURL(file));
            setPredictionData(null);
            setError(null);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError("Please select an image first.");
            return;
        }

        setIsLoading(true);
        setError(null);
        setPredictionData(null);

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await axios.post("http://127.0.0.1:8000/predict-image", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });
            setPredictionData(response.data);
        } catch (err) {
            console.error("Error uploading file:", err);
            setError("An error occurred during analysis. Please check the backend server and try again.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div style={{ maxWidth: '800px', margin: '20px auto', fontFamily: 'Arial, sans-serif', textAlign: 'center' }}>
            <h2>Bovine Image Analysis</h2>
            <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'block', margin: '20px auto' }} />

            <button onClick={handleUpload} disabled={isLoading} style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}>
                {isLoading ? 'Analyzing...' : 'Analyze Image'}
            </button>

            {error && <p style={{ color: 'red', marginTop: '20px' }}>{error}</p>}

            <div style={{ marginTop: '30px', display: 'flex', gap: '20px', alignItems: 'flex-start', justifyContent: 'center' }}>
                {previewImage && (
                    <div style={{ flex: 1, maxWidth: '350px' }}>
                        <h3>Image Preview</h3>
                        <img src={previewImage} alt="Selected Preview" style={{ maxWidth: '100%', borderRadius: '8px' }} />
                    </div>
                )}

                {isLoading && <p>Loading results...</p>}
                {predictionData && (
                    <div style={{ flex: 1, maxWidth: '350px', textAlign: 'left', background: '#f9f9f9', padding: '20px', borderRadius: '8px' }}>
                        <h3>Analysis Results</h3>
                        <p><strong>Animal Type:</strong> {predictionData.prediction.animal_type}</p>
                        <p><strong>Predicted Breed:</strong> {predictionData.prediction.breed}</p>
                        <p><strong>Visual Health Status:</strong> {predictionData.prediction.health_status}</p>
                        <hr />
                        <h4>Breed Information</h4>
                        <p><strong>Milk Yield:</strong> {predictionData.breed_info.milk_yield}</p>
                        <p><strong>Weight Range:</strong> {predictionData.breed_info.weight_range}</p>
                        <p><i>{predictionData.breed_info.info}</i></p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ImageUploader;
