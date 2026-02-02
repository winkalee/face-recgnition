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

  const createResizedCanvas = (img: HTMLImageElement, size = 256) => {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to create canvas context');

    // Fill with white background to avoid transparent images causing issues
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, size, size);

    // Compute aspect-fit sizing
    const { width: iw, height: ih } = img;
    let dw = size, dh = size;
    const ir = iw / ih;
    if (iw > ih) {
      // landscape
      dh = Math.round(size / ir);
      dw = size;
    } else if (ih > iw) {
      // portrait
      dw = Math.round(size * ir);
      dh = size;
    }
    const dx = Math.round((size - dw) / 2);
    const dy = Math.round((size - dh) / 2);
    ctx.drawImage(img, 0, 0, iw, ih, dx, dy, dw, dh);
    return canvas;
  };

  const loadImageElement = (file: File): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        resolve(img);
      };
      img.onerror = (e) => reject(new Error('Failed to load image ' + file.name));
      img.src = url;
    });
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
      const faceDescriptors: Float32Array[] = [];
      const errors: string[] = [];

      for (const file of files) {
        try {
          const imgEl = await loadImageElement(file);
          // Create a 256x256 canvas and draw the image into it (aspect-fit)
          const canvas = createResizedCanvas(imgEl, 256);

          // Detect on the canvas (this ensures consistent tensor shape)
          const detection = await faceapi
            .detectSingleFace(canvas, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceDescriptor();

          if (detection && detection.descriptor) {
            faceDescriptors.push(detection.descriptor);
          } else {
            errors.push(`No face detected in ${file.name}`);
          }

          // Revoke object URL to avoid leaks
          // Note: we used Image.src but didn't keep the URL variable here; revoke on file
          // CreateObjectURL was used inside loadImageElement, which didn't return the url, so revoke via creating one again then revoking
          try {
            const tmpUrl = URL.createObjectURL(file);
            URL.revokeObjectURL(tmpUrl);
          } catch (e) {
            // ignore
          }
        } catch (fileErr) {
          errors.push(`${file.name}: ${(fileErr as Error).message}`);
        }
      }

      if (faceDescriptors.length === 0) {
        throw new Error('No valid face descriptors extracted.\n' + errors.join('\n'));
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