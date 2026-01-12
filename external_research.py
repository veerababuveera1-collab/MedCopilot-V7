import streamlit as st
from groq import Groq

def external_research_answer(query):

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a hospital-grade medical research AI. Provide evidence-based medical information."},
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        max_tokens=800
    )

    return {
        "answer": response.choices[0].message.content
    }
