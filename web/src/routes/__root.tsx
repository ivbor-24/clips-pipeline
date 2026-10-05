import { createRootRoute, Navigate, Outlet, useRouterState } from '@tanstack/react-router';
import { AuthProvider, useAuth } from '@/features/auth/AuthContext';

export const Route = createRootRoute({
  component: () => (
    <AuthProvider>
      <AuthGate />
    </AuthProvider>
  ),
});

// The login page is shown only when the shared password is on and there is
// no valid session; otherwise it is skipped.
function AuthGate() {
  const { session, loading } = useAuth();
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  if (loading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  if (!session && pathname !== '/login') return <Navigate to="/login" />;
  if (session && pathname === '/login') return <Navigate to="/dashboard" />;
  return <Outlet />;
}
