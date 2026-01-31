"use client";

import { useState, useRef } from 'react';
import * as faceapi from 'face-api.js';

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [description, setDescription] = useState('');
  const [inferenceDesc, setInferenceDesc] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files).slice(0, 3);
      setFiles(selectedFiles);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) return;

    setLoading(true);
    try {
      // Initialize TensorFlow backend
      await faceapi.tf.setBackend('webgl');
      await faceapi.tf.ready();

      // Load face-api models (assuming models are in public/models)
      await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
      await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
      await faceapi.nets.faceRecognitionNet.loadFromUri('/models');

      // Process faces
      const faceDescriptors = [];
      for (const file of files) {
        const img = await faceapi.fetchImage(URL.createObjectURL(file));
        const detection = await faceapi.detectSingleFace(img, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptor();
        if (detection) {
          faceDescriptors.push(detection.descriptor);
        }
      }

      // Simulate search
      const mockResults = [
        { platform: 'Twitter', username: '@example1', post: 'Saw this person at the park.', similarity: 0.95 },
        { platform: 'Facebook', username: 'John Doe', post: 'Family reunion photo.', similarity: 0.88 },
        { platform: 'Instagram', username: '@user2', post: 'Vacation pic.', similarity: 0.92 },
      ];

      setResults(mockResults);
    } catch (error) {
      alert('Error processing images: ' + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleInference = () => {
    // Simulate inference
    const inferred = `Based on description: ${inferenceDesc}, possible changes: aged 10 years, changed hairstyle.`;
    alert(inferred);
  };

  return (
    <div className="min-h-screen bg-gray-100 py-8 px-4">
      <div className="max-w-2xl mx-auto bg-white p-6 rounded-lg shadow-md">
        <h1 className="text-2xl font-bold mb-4">Face Recognition Search</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Upload Photos (1-3)</label>
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileChange}
              ref={fileInputRef}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full p-2 border rounded"
              placeholder="Describe the person..."
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600 disabled:opacity-50"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-2">Face Inference</h2>
          <textarea
            value={inferenceDesc}
            onChange={(e) => setInferenceDesc(e.target.value)}
            className="w-full p-2 border rounded mb-2"
            placeholder="Describe possible changes..."
          />
          <button
            onClick={handleInference}
            className="bg-green-500 text-white py-2 px-4 rounded hover:bg-green-600"
          >
            Infer Changes
          </button>
        </div>
        {files.length > 0 && (
          <div className="mt-4">
            <h3 className="text-lg font-medium mb-2">Uploaded Photos</h3>
            <div className="flex space-x-2">
              {files.map((file, index) => (
                <img key={index} src={URL.createObjectURL(file)} alt={`Uploaded ${index}`} className="w-20 h-20 object-cover rounded" />
              ))}
            </div>
          </div>
        )}
        {results.length > 0 && (
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Search Results</h3>
            <ul className="space-y-2">
              {results.map((result, index) => (
                <li key={index} className="border p-2 rounded">
                  <strong>{result.platform} - {result.username}</strong>: {result.post} (Similarity: {result.similarity})
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
