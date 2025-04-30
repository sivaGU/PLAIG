import streamlit as st
from streamlit import session_state as ss


if "sidebar_state" not in ss:
    ss.sidebar_state = "expanded"
st.set_page_config(page_title="Multipage App", initial_sidebar_state=ss.sidebar_state)
st.markdown("<h1 style='text-align: center;'>Welcome to PLAIG's Documentation Webpage</h1>", unsafe_allow_html=True)
# if "sidebar_visible" not in st.session_state:
#     st.session_state.sidebar_visible = "expanded"
# if st.button("Toggle Sidebar"):
#     st.session_state.sidebar_visible = ("collapsed" if st.session_state.sidebar_visible == "expanded" else "expanded")
st.markdown("<p style='text-align: center; font-size: 24px'>PLAIG is a GNN-based deep learning model for protein-ligand binding affinity "
            "prediction. This app provides documentation on how to use PLAIG and details how PLAIG generates "
            "graph representations to predict binding affinity. By clicking on the tabs in the side bar, "
            "you will be able to read documentation and test PLAIG's binding affinity prediction model in three "
            "different ways. Please read the citation listed below for background information on "
            "PLAIG before navigating through this webpage.</p>", unsafe_allow_html=True)
st.image("GNN Model Framework.png")
st.markdown("<p style='text-align: center; font-size: 20px'>Flowchart for PLAIG's Model Framework</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size: 18px'><b>Citation:</b><br>Samudrala, M. V.; Dandibhotla, S.; Kaneriya, A.; Dakshanamurthy, S. PLAIG: Protein–Ligand Binding Affinity Prediction Using a Novel Interaction-Based Graph Neural Network Framework. ACS Bio Med Chem Au 2025. https://doi.org/10.1021/acsbiomedchemau.5c00053.</p>", unsafe_allow_html=True)
st.markdown("[Click here to access publication](https://pubs.acs.org/doi/10.1021/acsbiomedchemau.5c00053)")



