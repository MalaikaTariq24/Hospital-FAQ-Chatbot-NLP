import streamlit as st
import joblib
import re
import pandas as pd
import nltk

# -----------------------------------------------------------------------------
# 1. NLTK Setup (Fast & Zero-Hang)
# -----------------------------------------------------------------------------
for pkg in ['stopwords', 'punkt']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords

# -----------------------------------------------------------------------------
# 2. Page Configuration & Modern Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hospital AI Triage System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #FAFAFA;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    
    /* Result Badges & Cards */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 10px;
    }
    .badge-category {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1rem;
        border: 1px solid #BFDBFE;
    }
    .badge-routine {
        background-color: #F0FDF4;
        color: #15803D;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1rem;
        border: 1px solid #BBF7D0;
    }
    .badge-emergency {
        background-color: #FEF2F2;
        color: #B91C1C;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1rem;
        border: 1px solid #FECACA;
    }

    /* Modern Attractive Action Response Card */
    .action-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        border-left: 6px solid #0EA5E9;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .action-card.emergency-card {
        border-left: 6px solid #EF4444;
        background: #FEF2F2;
    }
    .card-header-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 20px;
        margin-bottom: 10px;
    }
    .info-badge {
        background-color: #E0F2FE;
        color: #0369A1;
    }
    .emergency-badge {
        background-color: #FEE2E2;
        color: #B91C1C;
    }
    .card-body-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 12px;
    }
    
    /* Interactive Timing & Info Grid */
    .timing-grid {
        display: flex;
        gap: 15px;
        margin-top: 10px;
        flex-wrap: wrap;
    }
    .timing-box {
        flex: 1;
        min-width: 210px;
        background: #F8FAFC;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    .timing-label {
        display: block;
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .timing-value {
        font-size: 1.05rem;
        color: #0F172A;
        font-weight: 700;
    }
    .highlight-contact {
        background-color: #0F172A;
        color: #0EA5E9;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: monospace;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Model & Artifacts Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model_cat = joblib.load('model_category.pkl')
        model_urg = joblib.load('model_urgency.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model_cat, model_urg, vectorizer, None
    except Exception as e:
        return None, None, None, str(e)

model_cat, model_urg, vectorizer, load_error = load_artifacts()

try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered)

def get_automated_response(category: str, urgency: str) -> str:
    """Returns styled HTML cards for maximum visual impact."""
    if urgency == "EMERGENCY":
        return """
        <div class="action-card emergency-card">
            <div class="card-header-badge emergency-badge">🚨 IMMEDIATE TRIAGE REQUIRED</div>
            <div class="card-body-title">Critical Care & Emergency Protocol</div>
            <div class="timing-grid">
                <div class="timing-box" style="background:#FFFFFF;">
                    <span class="timing-label">📍 Immediate Action</span>
                    <span class="timing-value" style="color:#B91C1C;">Direct Patient to Emergency Bay 1</span>
                </div>
                <div class="timing-box" style="background:#FFFFFF;">
                    <span class="timing-label">📞 Emergency Helpline</span>
                    <span class="timing-value" style="color:#B91C1C;">+92-42-111-911-911</span>
                </div>
            </div>
        </div>
        """
    
    if category == "VISITING_HOURS":
        return """
        <div class="action-card">
            <div class="card-header-badge info-badge">🕒 DEPARTMENT VISITING SCHEDULE</div>
            <div class="card-body-title">Inpatient & Ward Visiting Hours</div>
            <div class="timing-grid">
                <div class="timing-box">
                    <span class="timing-label">🏥 General Wards Schedule</span>
                    <span class="timing-value">04:00 PM – 07:00 PM <small style="color:#64748B; font-weight:normal;">(Daily)</small></span>
                </div>
                <div class="timing-box">
                    <span class="timing-label">🩺 ICU & Critical Care</span>
                    <span class="timing-value">Max 1 Attendant <small style="color:#64748B; font-weight:normal;">(Pass Required)</small></span>
                </div>
            </div>
        </div>
        """
        
    responses = {
        "APPOINTMENT": """
        <div class="action-card">
            <div class="card-header-badge info-badge">📅 APPOINTMENT GUIDANCE</div>
            <div class="card-body-title">OPD Specialist Booking</div>
            <div class="timing-grid">
                <div class="timing-box">
                    <span class="timing-label">📍 Physical Counter</span>
                    <span class="timing-value">OPD Hall - Counter 1</span>
                </div>
                <div class="timing-box">
                    <span class="timing-label">🌐 Portal</span>
                    <span class="timing-value">appointments.hospital.org</span>
                </div>
            </div>
        </div>
        """,
        "BILLING": """
        <div class="action-card">
            <div class="card-header-badge info-badge">💳 BILLING & ACCOUNTS</div>
            <div class="card-body-title">Patient Payment Counter</div>
            <div class="timing-grid">
                <div class="timing-box">
                    <span class="timing-label">📍 Counter Location</span>
                    <span class="timing-value">Cashier Desk - Counter 4</span>
                </div>
                <div class="timing-box">
                    <span class="timing-label">🕒 Operating Hours</span>
                    <span class="timing-value">24 Hours Active</span>
                </div>
            </div>
        </div>
        """,
        "PARKING": """
        <div class="action-card">
            <div class="card-header-badge info-badge">🅿️ PARKING FACILITIES</div>
            <div class="card-body-title">Visitor Parking Guidelines</div>
            <div class="timing-grid">
                <div class="timing-box">
                    <span class="timing-label">🚗 Free Visitor Parking</span>
                    <span class="timing-value">First 2 Hours Free</span>
                </div>
                <div class="timing-box">
                    <span class="timing-label">🎫 Overnight Passes</span>
                    <span class="timing-value">Available at Main Gate</span>
                </div>
            </div>
        </div>
        """,
        "PHARMACY": """
        <div class="action-card">
            <div class="card-header-badge info-badge">💊 PHARMACY SERVICES</div>
            <div class="card-body-title">Central Medicine Desk</div>
            <div class="timing-grid">
                <div class="timing-box">
                    <span class="timing-label">📍 Location</span>
                    <span class="timing-value">Ground Floor (Near Main Lobby)</span>
                </div>
                <div class="timing-box">
                    <span class="timing-label">🕒 Service Status</span>
                    <span class="timing-value">Open 24/7 Daily</span>
                </div>
            </div>
        </div>
        """,
        "MEDICAL_RECORDS": """
        <div class="action-card">
            <div class="card-header-badge info-badge">📑 DIAGNOSTIC REPORTS</div>
            <div class="card-body-title">Lab & Records Collection</div>
            <div class="timing-grid">
                <div class="timing-box">
                    <span class="timing-label">📍 Collection Counter</span>
                    <span class="timing-value">Reports Desk - Counter 2</span>
                </div>
                <div class="timing-box">
                    <span class="timing-label">📲 Digital Download</span>
                    <span class="timing-value">SMS Login Portal</span>
                </div>
            </div>
        </div>
        """
    }
    
    return responses.get(category, """
    <div class="action-card">
        <div class="card-header-badge info-badge">ℹ️ GENERAL HELPDESK</div>
        <div class="card-body-title">Patient Support Services</div>
        <p style="color:#475569; margin:0;">Query routed to Central Information Desk in Main Lobby for further assistance.</p>
    </div>
    """)

# -----------------------------------------------------------------------------
# 4. Sidebar UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital-2.png", width=60)
    st.markdown("### Hospital AI Routing")
    st.caption("NLP Dual-Task Triage Engine")
    st.markdown("---")
    
    st.markdown("**System Metadata**")
    st.write("• **Algorithm:** Random Forest")
    st.write("• **Vectorization:** TF-IDF")
    st.write("• **Dataset:** 2,100 Records")
    st.write("• **Intent Accuracy:** 100.00%")
    st.write("• **Urgency Accuracy:** 87.50%")
    
    st.markdown("---")
    st.caption("Designed for Academic & Clinical Demonstration")

# -----------------------------------------------------------------------------
# 5. Main Interface
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🏥 Hospital FAQ & Urgency Triage Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated natural language processing system for patient query classification and priority determination.</div>', unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ **Artifact Error:** {load_error}")
    st.info("Ensure `model_category.pkl`, `model_urgency.pkl`, and `vectorizer.pkl` are located in the project folder.")
else:
    c1, c2 = st.columns([3, 2])
    
    with c2:
        sample_choice = st.selectbox(
            "Quick Sample Selection:",
            [
                "-- Select a sample --",
                "Emergency! Someone is having severe chest pain near OPD!",
                "I need to book an appointment with a cardiologist tomorrow.",
                "Can I pay my treatment bill through an online mobile app?",
                "ICU visiting timing kya hain aaj?",
                "Is basement parking free for patient attendees overnight?",
                "Where can I collect my blood lab report chart?"
            ]
        )
    
    default_query = "" if sample_choice.startswith("--") else sample_choice
    
    with c1:
        user_query = st.text_input(
            "Enter Patient Query:",
            value=default_query,
            placeholder="e.g., Doctor se appointment book krni hai time bta do."
        )

    btn_submit = st.button("🔍 Analyze Query", type="primary", use_container_width=True)

    if btn_submit or (user_query and not sample_choice.startswith("--")):
        if not user_query.strip():
            st.warning("Please enter a text query to process.")
        else:
            with st.spinner("Processing pipeline..."):
                cleaned = clean_text(user_query)
                vec = vectorizer.transform([cleaned])
                
                cat_pred = str(model_cat.predict(vec)[0]).upper()
                urg_pred = str(model_urg.predict(vec)[0]).upper()

            st.markdown("---")
            st.markdown("##### 📋 Analysis Results")

            rc1, rc2 = st.columns(2)

            with rc1:
                st.markdown(
                    f'<div class="card">'
                    f'<small style="color:#64748B;">TARGET DEPARTMENT</small><br><br>'
                    f'<span class="badge-category">📍 {cat_pred}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with rc2:
                badge_style = "badge-emergency" if urg_pred == "EMERGENCY" else "badge-routine"
                icon = "🔴" if urg_pred == "EMERGENCY" else "🟢"
                
                st.markdown(
                    f'<div class="card">'
                    f'<small style="color:#64748B;">PRIORITY STATUS</small><br><br>'
                    f'<span class="{badge_style}">{icon} {urg_pred}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Prominent HTML Action Card Rendering
            auto_card_html = get_automated_response(cat_pred, urg_pred)
            st.markdown(auto_card_html, unsafe_allow_html=True)

            with st.expander("🛠️ View Processing Pipeline Details"):
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.write("**Raw Query:**", user_query)
                    st.write("**Cleaned Query:**", cleaned if cleaned else "(Only Stopwords)")
                with ec2:
                    st.write("**Feature Dimensions:**", f"{vec.shape}")
                    st.write("**Ensemble Model:**", "Dual Random Forest")
