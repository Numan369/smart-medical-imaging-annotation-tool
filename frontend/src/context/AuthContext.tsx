import React, { createContext, useContext, useState, useEffect } from "react";
import { UserProfile } from "../types";
import { authService, SignInCredentials, SignUpCredentials } from "../services/authService";

interface AuthContextType {
  user: UserProfile | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  signIn: (credentials: SignInCredentials) => Promise<UserProfile>;
  signUp: (credentials: SignUpCredentials) => Promise<UserProfile>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function initAuth() {
      try {
        const current = await authService.getCurrentUser();
        setUser(current);
      } catch (err) {
        console.error("Auth init error", err);
      } finally {
        setIsLoading(false);
      }
    }
    initAuth();
  }, []);

  const signIn = async (credentials: SignInCredentials) => {
    const loggedInUser = await authService.signIn(credentials);
    setUser(loggedInUser);
    return loggedInUser;
  };

  const signUp = async (credentials: SignUpCredentials) => {
    const newUser = await authService.signUp(credentials);
    setUser(newUser);
    return newUser;
  };

  const signOut = async () => {
    await authService.signOut();
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        signIn,
        signUp,
        signOut,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
