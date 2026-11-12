
import streamlit as st
import sqlite3
import hashlib
import secrets

from pathlib import Path


st.set_page_config(
    page_title="CattleCare AI | Registration",
    page_icon="🔐",
    layout="centered"
)


# ---------------- DATABASE ----------------

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "users.db"


def init_db():

    with sqlite3.connect(str(DB_PATH)) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL
            )
        """)


# ---------------- PASSWORD HASHING ----------------

def hash_password(password, salt):

    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        200000
    ).hex()


# ---------------- REGISTRATION ----------------

def register_user(name, email, password):

    salt = secrets.token_hex(16)

    password_hash = hash_password(password, salt)

    try:

        with sqlite3.connect(str(DB_PATH)) as conn:

            conn.execute(
                """
                INSERT INTO users
                (name, email, password_hash, salt)
                VALUES (?, ?, ?, ?)
                """,
                (
                    name.strip(),
                    email.strip().lower(),
                    password_hash,
                    salt
                )
            )

        return True

    except sqlite3.IntegrityError:
        return False


try:
    init_db()
except sqlite3.Error as exc:
    st.error(f"Database initialization failed: {exc}")
    st.stop()


# ---------------- CSS ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #f7fbf7);
    font-family: Arial, sans-serif;
}

.block-container {
    max-width: 550px;
    padding-top: 2rem;
}

.stTextInput label {
    color: #234b31;
    font-weight: 600;
}

.stFormSubmitButton > button,
.stButton > button {
    width: 100%;
    background: #21854b;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 45px;
    font-weight: bold;
}

.stFormSubmitButton > button:hover,
.stButton > button:hover {
    background: #176b39;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------- REGISTRATION FORM ----------------

st.markdown("## 🐄 Create Your Account")

st.write("Register to access cattle disease prediction.")

with st.form("registration_form"):

    name = st.text_input(
        "Full Name",
        placeholder="Enter your full name"
    )

    email = st.text_input(
        "Email Address",
        placeholder="Enter your email"
    )

    password = st.text_input(
        "Create Password",
        type="password",
        placeholder="At least 8 characters"
    )

    confirm_password = st.text_input(
        "Confirm Password",
        type="password",
        placeholder="Re-enter your password"
    )

    submitted = st.form_submit_button("Create Account")


# ---------------- VALIDATION ----------------

if submitted:

    clean_name = name.strip()
    clean_email = email.strip().lower()

    if not all([
        clean_name,
        clean_email,
        password,
        confirm_password
    ]):

        st.error("Please fill in all fields.")

    elif (
        "@" not in clean_email
        or "." not in clean_email.split("@")[-1]
        or any(char.isspace() for char in clean_email)
    ):

        st.error("Please enter a valid email address.")

    elif len(password) < 8:

        st.error("Password must contain at least 8 characters.")

    elif password != confirm_password:

        st.error("Passwords do not match.")

    else:

        if register_user(
            clean_name,
            clean_email,
            password
        ):

            st.success("Registration successful!")

            st.info("Your account has been created.")

            if st.button("Go to Login"):
                st.switch_page("app.py")

        else:

            st.error(
                "This email is already registered. Please log in."
            )


st.markdown("---")

if st.button("Back to Login"):
    st.switch_page("app.py")

st.caption("© 2026 CattleCare AI | Educational Project")
