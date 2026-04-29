import axios from 'axios';
import type { BlindDetectionResult, ComparativeResult } from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 60000,
});

export async function detectBlind(file: File): Promise<BlindDetectionResult> {
  const form = new FormData();
  form.append('image', file);
  const { data } = await api.post<BlindDetectionResult>('/detect/blind', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function detectComparative(
  original: File,
  suspected: File
): Promise<ComparativeResult> {
  const form = new FormData();
  form.append('original', original);
  form.append('suspected', suspected);
  const { data } = await api.post<ComparativeResult>('/detect/compare', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}
