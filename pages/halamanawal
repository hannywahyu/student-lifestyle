import streamlit as st

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Data Description", "Prediction", "About Naive Bayes"])
# Page 1: Data description
if page == "Data Description":
    st.title("Data Description")
    
    # Input for name
    name = st.text_input("Enter your name:", "")
    
    # Slider for age
    age = st.slider("Select your age:", min_value=0, max_value=100, value=25)
    
    if name:
        st.write(f"Hello, **{name}**! You are **{age}** years old.")
    else:
        st.write("Please enter your name.")

# Page 2: Prediction
elif page == "Prediction":
    st.title("Prediction")
    st.write("This is the prediction page")

# Page 3: About Naive Bayes
elif page == "About Naive Bayes":
    st.title("About Naive Bayes")
    st.write("""
        Naive Bayes is a classification technique based on Bayes' Theorem with an assumption of independence among predictors.
        It works well with large datasets and is particularly suited for text classification problems such as spam detection.
    """)
