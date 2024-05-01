import pickle
import pandas as pd
import streamlit as st

model = pickle.load(open('model.pkl', 'rb'))


def predict(age, sex, chest_pain_type, resting_bp_s, cholesterol,
            fasting_blood_sugar, resting_ecg, max_heart_rate,
            exercise_angina, old_peak, ST_slope):
    predictions = model.predict(pd.DataFrame([[age, sex, chest_pain_type, resting_bp_s, cholesterol,
                                               fasting_blood_sugar, resting_ecg, max_heart_rate,
                                               exercise_angina, old_peak, ST_slope]],
                                             columns=['age', 'sex', 'chest pain type', 'resting bp s',
                                                      'cholesterol', 'fasting blood sugar', 'resting ecg',
                                                      'max heart rate',
                                                      'exercise angina', 'oldpeak', 'ST slope']))

    return predictions


st.title('Приложение для определения вероятности заболеваний сердца')
st.image('f81806ce3c87fb44eac81b007f833dd2-1024x576.jpg')
st.header('Для определение вероятности заболеваний сердца ответьте на несколько вопросов :')
# Input text
age = st.number_input('возраст пациента в годах :', min_value=0, max_value=200, value=1)
sex = st.selectbox('пол пациента 1 - мужской, 0- женский:', [0, 1])
chest_pain_type = st.selectbox('боль в груди : 1 - типичная для стенокардии, 2- нетипичная для стенокардии, 3- боль '
                               'несвязанная со стенокардией, 4- отсутствие боли в груди', [1, 2, 3, 4])
resting_bp_s = st.number_input('AД систолическое в покое в мм.рт.ст:', min_value=0, max_value=300, value=1)
cholesterol = st.number_input('холестерин в mg/dl:', min_value=0, max_value=300, value=1)
fasting_blood_sugar = st.selectbox('глюкоза натощак 1 - повышено , 0 - норма :', [0, 1])
resting_ecg = st.selectbox('ЭКГ в покое 0 - норма, 1- наличие аномалии ST-T (инверсия зубца T),элевация или депрессия '
                           'ST > 0,05 мВ 2- вероятная или достоверная левожелудочковая недостаточность, гипертрофия '
                           'по критериям Эстеса', [0, 1, 2])
max_heart_rate = st.number_input('ЧCC max: ', min_value=1, max_value=200, value=1)
exercise_angina = st.selectbox('симптомы стенокардии при нагрузке 0- нет. 1- да: ', [0, 1])
old_peak = st.number_input('old_peak(измеряется как вертикальное расстояние от изоэлектрической линии до самой '
                           'высокой точки смещения сегмента ST) в мм:', min_value=0.0, max_value=5.0, step=0.1)
ST_slope = st.selectbox('Наклон сегмента ST при пиковой нагрузке 1- положительный наклон ST 2- горизонтальный наклон '
                        'ST 3-отрицательный наклон ST ', [1, 2, 3])
if st.button('определить вероятность'):
    p = predict(age, sex, chest_pain_type, resting_bp_s, cholesterol,
                fasting_blood_sugar, resting_ecg, max_heart_rate,
                exercise_angina, old_peak, ST_slope)
    if p == [0]:
        st.success(f'вероятность :низкая')
    else:
        st.success(f'вероятность :высокая')
