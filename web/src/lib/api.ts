import axios from 'axios';

// Base URL for REST calls and EventSource/<video> URLs. In dev, the Vite
// proxy forwards /api to the backend; override with VITE_API_URL if needed.
export const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// A 401 means the shared password is on (API_PASSWORD) and the session is
// missing or expired. Without a password the API never answers 401.
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // /auth/me and /auth/login report to AuthContext and the login form;
      // redirecting here would reload /login in a loop.
      const url: string = error.config?.url || '';
      const isAuthRequest = url.includes('/auth/login') || url.includes('/auth/me');
      if (!isAuthRequest) {
        localStorage.removeItem('access_token');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// EventSource, <video> and download links cannot send headers: pass the
// session token in the URL when there is one.
export function withToken(url: string): string {
  const token = localStorage.getItem('access_token');
  if (!token) return url;
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}token=${encodeURIComponent(token)}`;
}
