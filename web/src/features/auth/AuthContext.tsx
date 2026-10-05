import { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import { api } from '@/lib/api';

// One built-in user. Without API_PASSWORD the session always
// exists; with it, the shared password is exchanged for a session token.
interface Session {
  username: string;
  password_required: boolean;
}

interface AuthContextType {
  session: Session | null;
  loading: boolean;
  login: (password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get('/auth/me')
      .then((res) => {
        setSession(res.data);
        // A token from a time the password was on is no longer needed.
        if (!res.data.password_required) localStorage.removeItem('access_token');
      })
      .catch(() => localStorage.removeItem('access_token'))
      .finally(() => setLoading(false));
  }, []);

  const login = async (password: string) => {
    const res = await api.post('/auth/login', { password });
    localStorage.setItem('access_token', res.data.access_token);
    const me = await api.get('/auth/me');
    setSession(me.data);
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    setSession(null);
  };

  return (
    <AuthContext.Provider value={{ session, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
}
