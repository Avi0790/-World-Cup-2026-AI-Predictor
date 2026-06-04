"""
FIFA Oracle — Firebase Authentication & Firestore Integration
Handles user registration, login, and user data storage via Firebase.
"""

import streamlit as st
import requests
import json
import os

# ── Firebase Config ───────────────────────────────────────────────────────────
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyAgRTQG_c2czPYu385oRot44Bhx1APpR1s",
    "authDomain": "fifa-world-cup-prediction.firebaseapp.com",
    "projectId": "fifa-world-cup-prediction",
    "storageBucket": "fifa-world-cup-prediction.firebasestorage.app",
    "messagingSenderId": "720056258849",
    "appId": "1:720056258849:web:9ed6fe4ded66d6dd812e60",
}

# Firebase REST API endpoints
FIREBASE_API_KEY = FIREBASE_CONFIG["apiKey"]
SIGN_UP_URL   = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
SIGN_IN_URL   = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
USER_INFO_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_API_KEY}"

# Firestore REST API
FIRESTORE_URL = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_CONFIG['projectId']}/databases/(default)/documents"


def firebase_sign_up(email, password, username):
    """Register a new user with Firebase Auth."""
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    try:
        r = requests.post(SIGN_UP_URL, json=payload)
        data = r.json()
        if "error" in data:
            msg = data["error"]["message"]
            if msg == "EMAIL_EXISTS":
                return False, "This email is already registered."
            elif msg == "WEAK_PASSWORD : Password should be at least 6 characters":
                return False, "Password must be at least 6 characters."
            else:
                return False, f"Registration error: {msg}"

        # Save user profile to Firestore
        uid = data["localId"]
        id_token = data["idToken"]
        save_user_profile(uid, username, email, id_token)
        return True, {"uid": uid, "email": email, "username": username, "idToken": id_token}

    except Exception as e:
        return False, f"Connection error: {str(e)}"


def firebase_sign_in(email, password):
    """Sign in existing user with Firebase Auth."""
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    try:
        r = requests.post(SIGN_IN_URL, json=payload)
        data = r.json()
        if "error" in data:
            msg = data["error"]["message"]
            if msg == "EMAIL_NOT_FOUND":
                return False, "No account found with this email."
            elif msg == "INVALID_PASSWORD":
                return False, "Incorrect password."
            elif "INVALID_LOGIN_CREDENTIALS" in msg:
                return False, "Invalid email or password."
            else:
                return False, f"Login error: {msg}"

        uid      = data["localId"]
        id_token = data["idToken"]

        # Get username from Firestore
        profile = get_user_profile(uid, id_token)
        username = profile.get("username", email.split("@")[0])

        return True, {
            "uid":      uid,
            "email":    email,
            "username": username,
            "idToken":  id_token,
        }

    except Exception as e:
        return False, f"Connection error: {str(e)}"


def save_user_profile(uid, username, email, id_token):
    """Save user profile document to Firestore."""
    url = f"{FIRESTORE_URL}/users/{uid}"
    headers = {"Authorization": f"Bearer {id_token}"}
    payload = {
        "fields": {
            "username":    {"stringValue": username},
            "email":       {"stringValue": email},
            "uid":         {"stringValue": uid},
            "joined":      {"stringValue": "2026"},
            "predictions": {"integerValue": "0"},
        }
    }
    try:
        requests.patch(url, json=payload, headers=headers)
    except Exception:
        pass  # Non-critical — auth still works


def get_user_profile(uid, id_token):
    """Fetch user profile from Firestore."""
    url = f"{FIRESTORE_URL}/users/{uid}"
    headers = {"Authorization": f"Bearer {id_token}"}
    try:
        r = requests.get(url, headers=headers)
        data = r.json()
        if "fields" in data:
            return {k: list(v.values())[0] for k, v in data["fields"].items()}
    except Exception:
        pass
    return {}


def increment_predictions(uid, id_token, current_count):
    """Increment user's prediction count in Firestore."""
    url = f"{FIRESTORE_URL}/users/{uid}"
    headers = {"Authorization": f"Bearer {id_token}"}
    payload = {
        "fields": {
            "predictions": {"integerValue": str(current_count + 1)}
        }
    }
    try:
        requests.patch(url, json=payload, headers=headers,
                       params={"updateMask.fieldPaths": "predictions"})
    except Exception:
        pass


def save_prediction(uid, id_token, team_a, team_b, stage, result):
    """Save a prediction to Firestore under the user's predictions subcollection."""
    import time
    pred_id = str(int(time.time()))
    url = f"{FIRESTORE_URL}/users/{uid}/predictions/{pred_id}"
    headers = {"Authorization": f"Bearer {id_token}"}
    payload = {
        "fields": {
            "team_a":      {"stringValue": team_a},
            "team_b":      {"stringValue": team_b},
            "stage":       {"stringValue": stage},
            "prob_a_win":  {"doubleValue": float(result["prob_a_win"])},
            "prob_b_win":  {"doubleValue": float(result["prob_b_win"])},
            "prob_draw":   {"doubleValue": float(result["prob_draw"])},
            "score":       {"stringValue": result["predicted_score"]},
            "confidence":  {"doubleValue": float(result["confidence"])},
            "timestamp":   {"stringValue": pred_id},
        }
    }
    try:
        r = requests.patch(url, json=payload, headers=headers)
        return r.status_code in [200, 201]
    except Exception:
        return False


def get_user_predictions(uid, id_token):
    """Fetch all predictions for a user from Firestore."""
    url = f"{FIRESTORE_URL}/users/{uid}/predictions"
    headers = {"Authorization": f"Bearer {id_token}"}
    try:
        r = requests.get(url, headers=headers)
        data = r.json()
        preds = []
        for doc in data.get("documents", []):
            fields = doc.get("fields", {})
            preds.append({k: list(v.values())[0] for k, v in fields.items()})
        return sorted(preds, key=lambda x: x.get("timestamp",""), reverse=True)
    except Exception:
        return []
