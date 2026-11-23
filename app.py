import joblib
import numpy as np
import pandas as pd
import streamlit as st

obj = joblib.load("model/aussie_rain_logistic.joblib")

model = obj["model"]
imputer = obj["imputer"]
scaler = obj["scaler"]
encoder = obj["encoder"]

numeric_cols = obj["numeric_cols"]
categorical_cols = obj["categorical_cols"]
encoded_cols = obj["encoded_cols"]


def predict(single_input):
  input_df = pd.DataFrame([single_input])
  input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
  input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
  input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
  X_input = input_df[numeric_cols + encoded_cols]
  pred = model.predict(X_input)[0]
  probs = model.predict_proba(X_input)[0]
  prob = float(probs[list(model.classes_).index(pred)])
  all_probs = {str(k): round(float(v), 4) for k, v in zip(model.classes_, probs)}
  return pred, prob, all_probs


def num_field(col, use_slider=False, step=0.1):
  i = numeric_cols.index(col)
  low, high = float(scaler.data_min_[i]), float(scaler.data_max_[i])
  mean_val = round(float(imputer.statistics_[i]), 2)
  mean_val = max(low, min(high, mean_val))
  if not st.checkbox(f"{col}", value=False):
    return np.nan
  if use_slider:
    return st.slider(
        col,
        min_value=low,
        max_value=high,
        value=mean_val,
        step=step,
        key=f"s_{col}",
    )
  return st.number_input(
      col,
      min_value=low,
      max_value=high,
      value=mean_val,
      step=step,
      key=f"n_{col}",
  )


st.title("Прогноз дощу в Австралії")
st.markdown(
    "Хочете дізнатися, чи падатиме завтра дощ?\n\n"
    "Заповніть відомі сьогоднішні показники погоди і **модель логістичної регресії** "
    "розрахує прогноз на завтра."
)

st.header("Введіть значення сьогоднішніх вимірів:")
col1, col2, col3, col4, col5 = st.columns(5)

data = {}

with col1:
  for col in categorical_cols:
    opts = tuple(encoder.categories_[categorical_cols.index(col)])
    val = st.selectbox(col, opts, index=None, key=f"c_{col}")
    data[col] = np.nan if val is None else val

with col2:
  for col in [
      "MinTemp",
      "MaxTemp",
      "Temp9am",
      "Temp3pm",
  ]:
    data[col] = num_field(col, use_slider=True, step=0.1)

with col3:
  for col in [
      "WindGustSpeed",
      "WindSpeed9am",
      "WindSpeed3pm",
  ]:
    data[col] = num_field(col, use_slider=True, step=0.5)

with col4:
  for col in [
      "Humidity9am",
      "Humidity3pm",
      "Cloud9am",
      "Cloud3pm",
  ]:
    data[col] = num_field(col, use_slider=True, step=0.5)

with col5:
  for col in ["Sunshine"]:
    data[col] = num_field(col, use_slider=True, step=0.5)
  for col in [
      "Rainfall",
      "Evaporation",
      "Pressure9am",
      "Pressure3pm",
  ]:
    data[col] = num_field(
        col, use_slider=False, step=0.2 if col == "Rainfall" else 0.1
    )
      
if st.button("Зробити прогноз"):
  pred, prob, all_probs = predict(data)
  if str(pred) == "No":
    st.success(f"Прогноз: Завтра опадів не передбачається (ймовірність: **{prob:.2%}**)")
    st.image('images/sun.jpg', width="stretch")
  else:
    st.warning(f"Прогноз: Очікуйте завтра дощ (ймовірність: **{prob:.2%}**)")
    st.image('images/rain.webp', width="stretch")
  