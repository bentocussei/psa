# 📊 Pulsar Social Angolano

Análise interativa das redes sociais angolanas para identificar padrões, causas e intensidade de manifestações.

🔗 **Acesse o projeto aqui:** [pulsarsocial.streamlit.app](https://pulsarsocial.streamlit.app/)

---

**Autor:** Bento Cussei - *Data Analytics Specialist | Data Scientist*

O **Pulsar Social Angolano** é uma aplicação interativa de análise de dados sociais, focada em compreender e monitorar o comportamento e as interações da sociedade angolana nas redes sociais.  
Atualmente, a plataforma processa dados do **Instagram**, analisando publicações, comentários e interações para gerar insights **descritivos, diagnósticos e, futuramente, preditivos**.

---

## 🎯 Objetivo

Utilizar dados gerados nas redes sociais para:
- Compreender as **causas** e a **intensidade** de manifestações e eventos sociais.
- Identificar **atores influentes** e **temas mais discutidos**.
- Detectar **tendências emergentes** e padrões de engajamento.
- Disponibilizar análises de forma **visual e interativa** para todo o público.

---

## 📅 Escopo Atual (Fase 1)

A aplicação inclui:
- **Análise Descritiva**  
  Estatísticas gerais, métricas de engajamento, nuvem de palavras, distribuição temporal e rede de interações.
- **Análise Diagnóstica**  
  Modelagem de tópicos, análise de sentimentos e correlação entre variáveis.

> 🔮 **Fase 2**: Adição da **Análise Preditiva** quando os modelos estiverem maduros.

---

## 📑 Principais Funcionalidades

- **Principais métricas do dataset** (posts, comentários, likes, médias, período, redes sociais).
- **Top páginas e top posts por engajamento**.
- **Top comentários por engajamento**, incluindo link para o post original.
- **Distribuição temporal de postagens e interações**.
- **Análise de sentimentos**.
- **Nuvem de palavras** dinâmica.
- **Modelagem de tópicos**.
- **Rede de interações**.

---

## 🛠️ Tecnologias Utilizadas

- **Python**
- **Streamlit** – interface interativa.
- **Pandas / NumPy** – processamento de dados.
- **Plotly / Matplotlib** – visualização de dados.
- **WordCloud** – geração de nuvens de palavras.
- **PyArrow** – compatibilidade de dataframes.
- **NLTK / spaCy** – processamento de linguagem natural.
- **Scikit-learn** – modelagem de tópicos e análises.

---

## 📂 Estrutura do Projeto

```plaintext
pulsar-social-angolano/
│
├── data/                  # Dados brutos e processados
├── images/                # Capturas de tela e gráficos exportados
├── streamlit_app.py       # Código principal da aplicação
├── requirements.txt       # Dependências do projeto
└── README.md              # Este arquivo
```

---

## 🚀 Como Executar

1. **Clonar o repositório**
   ```bash
   git clone https://github.com/usuario/pulsar-social-angolano.git
   cd pulsar-social-angolano
   ```

2. **Criar ambiente virtual e instalar dependências**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

3. **Executar a aplicação**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Acessar no navegador**
   ```
   http://localhost:8501
   ```

---

## 📌 Observações

- Atualmente, a base de dados cobre **somente o Instagram**.
- As métricas principais no **Home** são pensadas para fácil compreensão do público geral, com textos explicativos para cada gráfico.
- O código está estruturado para permitir fácil expansão para outras redes sociais.