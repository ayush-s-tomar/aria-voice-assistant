# backend/services/auth.py
import os
import jwt
import httpx
from datetime import datetime, timedelta

GITHUB_CLIENT_ID     = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
JWT_SECRET           = os.getenv("JWT_SECRET", "fallback-secret")
JWT_ALGORITHM        = "HS256"
JWT_EXPIRE_DAYS      = 30


def get_github_auth_url() -> str:
    """Return the GitHub OAuth authorization URL."""
    return (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&scope=read:user"
    )


async def exchange_code_for_token(code: str) -> dict:
    """Exchange GitHub OAuth code for user info."""
    async with httpx.AsyncClient() as client:
        # Step 1: Get access token
        token_res = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id":     GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code":          code,
            },
            headers={"Accept": "application/json"},
        )
        token_data = token_res.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError("Failed to get access token from GitHub")

        # Step 2: Get user profile
        user_res = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user_data = user_res.json()

    return {
        "github_id": str(user_data["id"]),
        "username":  user_data["login"],
        "name":      user_data.get("name") or user_data["login"],
        "avatar":    user_data.get("avatar_url", ""),
    }


def create_jwt(user: dict) -> str:
    """Create a signed JWT for the user."""
    payload = {
        **user,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt(token: str) -> dict:
    """Verify and decode a JWT. Raises on invalid/expired."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])