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
  if not st.checkbox(f"{col} (сер. {mean_val})", value=False):
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
    "Хочете визначити чи буде завтра падати дощ? Заповніть необхідні дані та"
    " модель логістичної регресії дасть вам відповідь."
)

st.header("Введіть значення сьогоднішніх вимірів")
col1, col2, col3 = st.columns(3)

data = {}

with col1:
  for col in categorical_cols:
    opts = tuple(encoder.categories_[categorical_cols.index(col)])
    val = st.selectbox(col, opts, index=None, key=f"c_{col}")
    data[col] = np.nan if val is None else val

with col2:
  for col in [
      "WindGustSpeed",
      "WindSpeed9am",
      "WindSpeed3pm",
      "Humidity9am",
      "Humidity3pm",
      "Cloud9am",
      "Cloud3pm",
  ]:
    data[col] = num_field(col, use_slider=True, step=1.0)

with col3:
  for col in [
      "MinTemp",
      "MaxTemp",
      "Rainfall",
      "Evaporation",
      "Sunshine",
      "Pressure9am",
      "Pressure3pm",
      "Temp9am",
      "Temp3pm",
  ]:
    data[col] = num_field(
        col, use_slider=False, step=0.2 if col == "Rainfall" else 0.1
    )

if st.button("Визначити чи буде завтра дощ"):
  pred, prob, all_probs = predict(data)
  st.success(f"Прогноз: **{pred}** (ймовірність: **{prob:.2%}**)")
  st.write("Ймовірності по класах:", all_probs)