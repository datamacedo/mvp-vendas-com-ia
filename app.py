import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

st.header("💬 Faça uma pergunta em linguagem natural")

pergunta = st.text_input("Digite sua pergunta sobre os dados:")

if pergunta and st.button("Perguntar"):
    with st.spinner("Analisando com IA..."):
        api_key = st.secrets["openai_api_key"]
        llm = OpenAI(api_token=api_key)

        # 🔥 Adicione esta linha:
        df = pd.read_csv("vendas_200k.csv")

        sdf = SmartDataframe(df, config={"llm": llm})
        resposta = sdf.chat(pergunta)

        st.success("Resposta da IA:")
        st.write(resposta)
