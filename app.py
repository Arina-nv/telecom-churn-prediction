
import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Прогнозування відтоку клієнтів",
    page_icon="📊",
    layout="centered"
)


model = joblib.load("telecom_churn_model.joblib")


st.title("Прогнозування відтоку клієнтів")
st.write(
    "Введіть характеристики клієнта, щоб оцінити "
    "ймовірність припинення користування послугами."
)

st.info("Заповніть поля нижче.")


st.subheader("Характеристики клієнта")

is_tv_subscriber = st.selectbox(
    "Клієнт має підписку на телебачення?",
    options=[1, 0],
    format_func=lambda x: "Так" if x == 1 else "Ні"
)

is_movie_package_subscriber = st.selectbox(
    "Клієнт має пакет фільмів?",
    options=[1, 0],
    format_func=lambda x: "Так" if x == 1 else "Ні"
)

subscription_age = st.number_input(
    "Тривалість підписки (роки)",
    min_value=0.0,
    value=2.5,
    step=0.5
)

bill_avg = st.number_input(
    "Середній рахунок",
    min_value=0.0,
    value=20.0,
    step=1.0
)

reamining_contract = st.number_input(
    "Термін дії контракту за даними датасету",
    min_value=0.0,
    value=1.0,
    step=0.5,
    help="Вводьте значення у форматі, використаному в датасеті."
)

service_failure_count = st.number_input(
    "Кількість збоїв сервісу",
    min_value=0,
    value=0,
    step=1
)

download_avg = st.number_input(
    "Середній обсяг завантаження даних",
    min_value=0.0,
    value=10.0,
    step=1.0
)

upload_avg = st.number_input(
    "Середній обсяг передавання даних",
    min_value=0.0,
    value=2.0,
    step=0.5
)

download_over_limit = st.selectbox(
    "Перевищення ліміту завантаження?",
    options=[0, 1],
    format_func=lambda x: "Так" if x == 1 else "Ні"
)


if st.button("Прогнозувати відтік", type="primary"):

    
    new_customer = pd.DataFrame([{
        "is_tv_subscriber": is_tv_subscriber,
        "is_movie_package_subscriber": is_movie_package_subscriber,
        "subscription_age": subscription_age,
        "bill_avg": bill_avg,
        "reamining_contract": reamining_contract,
        "service_failure_count": service_failure_count,
        "download_avg": download_avg,
        "upload_avg": upload_avg,
        "download_over_limit": download_over_limit
    }])

    
    prediction = model.predict(new_customer)[0]
    probabilities = model.predict_proba(new_customer)[0]
    churn_probability = probabilities[list(model.classes_).index(1)]

    
    st.subheader("Результат прогнозування")

    st.metric(
        label="Оцінена ймовірність відтоку",
        value=f"{churn_probability:.2%}"
    )

    st.progress(float(churn_probability))

    if prediction == 1:
        st.error("Модель прогнозує підвищений ризик відтоку клієнта.")
    else:
        st.success("Модель прогнозує, що клієнт залишиться.")

    st.caption(
        "Це прогноз моделі, а не гарантія майбутньої поведінки клієнта."
    )
