import { UserProfile } from "../types";
import { getStoredUser, STORAGE_KEYS } from "../utils/storage";

export interface SignInCredentials {
  email: string;
  password?: string;
  rememberMe?: boolean;
}

export interface SignUpCredentials {
  name: string;
  email: string;
  password?: string;
  institution?: string;
}

/**
 * Service handling user authentication and profile simulation.
 * When integrating with a production backend, replace the localStorage operations
 * with calls to `POST /api/auth/login` and `POST /api/auth/register`.
 */
export const authService = {
  async getCurrentUser(): Promise<UserProfile | null> {
    const sessionToken = localStorage.getItem("smart_med_session_token");
    if (!sessionToken) return null;
    return getStoredUser();
  },

  async signIn(credentials: SignInCredentials): Promise<UserProfile> {
    // Simulated network delay
    await new Promise((res) => setTimeout(res, 450));

    let user = getStoredUser();
    if (credentials.email && credentials.email !== user.email) {
      user = {
        ...user,
        email: credentials.email,
        name: credentials.email.split("@")[0].replace(".", " ").toUpperCase(),
      };
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(user));
    }

    localStorage.setItem("smart_med_session_token", "mock-jwt-token-annotator");
    return user;
  },

  async signUp(credentials: SignUpCredentials): Promise<UserProfile> {
    await new Promise((res) => setTimeout(res, 550));

    const newUser: UserProfile = {
      id: `usr-${Date.now().toString().slice(-6)}`,
      name: credentials.name?.trim() || "Annotator",
      email: credentials.email?.trim(),
      role: "Annotator",
      institution: credentials.institution?.trim() || undefined,
    };

    localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(newUser));
    localStorage.setItem("smart_med_session_token", "mock-jwt-token-annotator");
    return newUser;
  },

  async signOut(): Promise<void> {
    localStorage.removeItem("smart_med_session_token");
  },
};
