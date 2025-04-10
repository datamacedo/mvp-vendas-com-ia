import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Analista de Dados Comerciais", layout="wide")
st.title("🤖 Analista de Dados Comerciais")

# ⬇️ Carregando os dados
df = pd.read_csv("vendas_200k.csv")
df['data_venda'] = pd.to_datetime(df['data_venda'])
df['ano_mes'] = df['data_venda'].dt.to_period('M').astype(str)

# 🔁 Histórico de perguntas
if "historico" not in st.session_state:
    st.session_state.historico = []

# 💬 Entrada da pergunta
pergunta = st.text_input("Digite sua pergunta sobre os dados de vendas:", key="pergunta")

if pergunta:
    pergunta_lower = pergunta.lower()
    api_key = st.secrets["openai_api_key"]
    llm = OpenAI(api_token=api_key)
    usar_ia = True

    if "mensal" in pergunta_lower or "comparativo" in pergunta_lower or "crescimento" in pergunta_lower:
        vendas_mensais = df.groupby('ano_mes')['valor'].sum().reset_index()
        vendas_mensais['variação_%'] = vendas_mensais['valor'].pct_change().fillna(0) * 100

        fig, ax = plt.subplots()
        ax.plot(vendas_mensais['ano_mes'], vendas_mensais['valor'], marker='o')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.dataframe(vendas_mensais)

        st.markdown(f"""
        ✅ **Mês com menor venda:** {vendas_mensais.loc[vendas_mensais['valor'].idxmin()]['ano_mes']}  
        🔼 **Maior crescimento:** {vendas_mensais.loc[vendas_mensais['variação_%'].idxmax()]['ano_mes']}
        """)
        resposta = "Análise mensal exibida com sucesso!"
        usar_ia = False

    elif "média" in pergunta_lower:
        media_geral = df['valor'].mean()
        st.metric(label="📊 Média Geral de Vendas", value=f"R$ {media_geral:,.2f}")
        resposta = f"A média geral das vendas é R$ {media_geral:,.2f}"
        usar_ia = False

    elif "top" in pergunta_lower or "mais vendidos" in pergunta_lower:
        top = df.groupby("modelo")["valor"].sum().sort_values(ascending=False).head(10)
        st.subheader("🏆 Top 10 Modelos Mais Vendidos")
        st.bar_chart(top)
        resposta = "Top 10 modelos mais vendidos exibidos!"
        usar_ia = False

    elif "meta" in pergunta_lower:
        vendas = df.groupby('ano_mes')['valor'].sum().reset_index()
        vendas['meta'] = 500000
        vendas['atingiu_meta'] = vendas['valor'] >= vendas['meta']
        st.dataframe(vendas)

        fig, ax = plt.subplots()
        ax.plot(vendas['ano_mes'], vendas['valor'], label="Realizado", marker='o')
        ax.plot(vendas['ano_mes'], vendas['meta'], label="Meta", linestyle='--')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        resposta = "Análise Meta vs Realizado gerada!"
        usar_ia = False

    elif "recomenda" in pergunta_lower:
        produto_recomendado = df.groupby("modelo")["valor"].sum().sort_values(ascending=False).index[0]
        resposta = f"💡 Recomendação: Invista mais no modelo **{produto_recomendado}**, ele tem o maior volume de vendas."
        st.success(resposta)
        usar_ia = False

    elif "previsão" in pergunta_lower:
        vendas = df.groupby('ano_mes')['valor'].sum().reset_index()
        vendas['indice'] = range(len(vendas))

        modelo = LinearRegression()
        modelo.fit(vendas[['indice']], vendas['valor'])

        futuro = pd.DataFrame({'indice': [len(vendas), len(vendas)+1, len(vendas)+2]})
        previsao = modelo.predict(futuro)

        st.subheader("🔮 Previsão para os próximos 3 meses:")
        for i, valor in enumerate(previsao):
            st.markdown(f"Mês {i+1}: **R$ {valor:,.2f}**")

        fig, ax = plt.subplots()
        ax.plot(vendas['ano_mes'], vendas['valor'], label="Histórico", marker='o')
        ax.plot(['+1', '+2', '+3'], previsao, label="Previsão", marker='x')
        ax.legend()
        st.pyplot(fig)

        resposta = "Previsão gerada com modelo linear simples."
        usar_ia = False

    # 🧠 IA como fallback
    if usar_ia:
        sdf = SmartDataframe(df, config={"llm": llm, "enable_plotting": True})
        resposta = sdf.chat(pergunta)
        st.success("Resposta da IA:")
        st.write(resposta)

    # 🗃️ Histórico
    st.session_state.historico.append((pergunta, resposta))

# 📜 Histórico
if st.session_state.historico:
    st.markdown("### 🗂️ Histórico de Perguntas")
    for i, (perg, resp) in enumerate(reversed(st.session_state.historico), 1):
        st.markdown(f"**{i}. {perg}**")
        st.write(resp)
