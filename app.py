
import sqlite3
import hashlib
import hmac
from pathlib import Path

import streamlit as st


# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CattleCare AI | Login",
    page_icon="🐄",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"


# ==================================================
# 2. DATABASE
# ==================================================

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


# ==================================================
# 3. PASSWORD HASHING
# ==================================================

def hash_password(password, salt):

    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        200000
    ).hex()


# ==================================================
# 4. AUTHENTICATION
# ==================================================

def authenticate_user(email, password):

    with sqlite3.connect(str(DB_PATH)) as conn:

        user = conn.execute(
            """
            SELECT name, email, password_hash, salt
            FROM users
            WHERE email = ?
            """,
            (email.strip().lower(),)
        ).fetchone()

    if user is None:
        return None

    name, saved_email, saved_hash, salt = user

    entered_hash = hash_password(password, salt)

    if hmac.compare_digest(entered_hash, saved_hash):

        return {
            "name": name,
            "email": saved_email
        }

    return None


# ==================================================
# 5. INITIALIZE DATABASE
# ==================================================

try:

    init_db()

except sqlite3.Error as exc:

    st.error(f"Database initialization failed: {exc}")
    st.stop()


# ==================================================
# 6. SESSION STATE
# ==================================================

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

if "user_name" not in st.session_state:
    st.session_state["user_name"] = None


# ==================================================
# 7. CSS DESIGN
# ==================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #e8f5e9, #f7fbf7);
}

.block-container {
    max-width: 550px;
    padding-top: 2rem;
}

.stButton > button,
.stFormSubmitButton > button {

    background-color: #21854b;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 44px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {

    background-color: #176b39;
    color: white;
    border: none;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 8. HEADER
# ==================================================

st.markdown("""""", unsafe_allow_html=True)


# ==================================================
# 9. LOGIN PAGE
# ==================================================

if not st.session_state["logged_in"]:

    st.markdown("## 🔐 Login to Your Account")

    with st.form("login_form"):

        email = st.text_input(
            "Email Address",
            placeholder="Enter your registered email"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password"
        )

        # Compatible with older Streamlit versions
        submitted = st.form_submit_button("Login")

    if submitted:

        if not email.strip() or not password:

            st.error("Please enter email and password.")

        else:

            user = authenticate_user(email, password)

            if user:

                st.session_state["logged_in"] = True
                st.session_state["user_email"] = user["email"]
                st.session_state["user_name"] = user["name"]

                st.success("Login successful!")

                st.info(
                    "Select Disease Prediction from the sidebar."
                )

            else:

                st.error("Invalid email or password.")

    st.markdown("---")

    st.write("Don't have an account yet?")

    if st.button("Create New Account"):

        st.info(
            "Select Create New Account from the sidebar."
        )


# ==================================================
# 10. LOGGED-IN PAGE
# ==================================================

else:

    st.success(
        f"Welcome, {st.session_state['user_name']}!"
    )

    st.markdown("### You are successfully logged in.")

    st.info(
        "Select Disease Prediction from the sidebar "
        "to analyze cattle images."
    )

    st.write(
        f"Registered Email: {st.session_state['user_email']}"
    )

    if st.button("Logout"):

        st.session_state["logged_in"] = False
        st.session_state["user_email"] = None
        st.session_state["user_name"] = None

        st.success("You have been logged out.")

        st.stop()


# ==================================================
# 11. FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "© 2026 CattleCare AI | Educational Project"
)
