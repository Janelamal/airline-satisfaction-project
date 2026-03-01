import streamlit as st
import pandas as pd
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml import Pipeline

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Airline Satisfaction System", layout="wide")
st.title("✈ Airline Passenger Satisfaction System")

# -----------------------------
# SPARK SESSION
# -----------------------------
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("AirlineProject") \
        .master("local[*]") \
        .getOrCreate()

spark = get_spark()

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = spark.read.csv(
        "airline_passenger_satisfaction.csv",
        header=True,
        inferSchema=True
    )

    if "_c0" in df.columns:
        df = df.drop("_c0")

    df = df.fillna({"arrival_delay_in_minutes": 0})

    df = df.withColumn(
        "label",
        when(col("satisfaction") == "satisfied", 1).otherwise(0)
    )

    return df

df = load_data()

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
page = st.sidebar.radio("Navigation", 
                        ["Dashboard", "Prediction"])

# =============================
# DASHBOARD PAGE
# =============================
if page == "Dashboard":
    st.subheader("Dataset Overview")

    st.write("Sample Data:")
    st.dataframe(df.limit(10).toPandas())

    st.subheader("Satisfaction Distribution")

    pandas_df = df.select("satisfaction").toPandas()
    st.bar_chart(pandas_df["satisfaction"].value_counts())

# =============================
# PREDICTION PAGE
# =============================
elif page == "Prediction":

    st.subheader("Predict Passenger Satisfaction")

    # Input fields
    wifi = st.slider("WiFi Service", 0, 5, 3)
    seat = st.slider("Seat Comfort", 0, 5, 3)
    food = st.slider("Food Quality", 0, 5, 3)
    delay = st.slider("Arrival Delay (minutes)", 0, 300, 30)

    if st.button("Predict"):

        # Prepare minimal dataset for prediction
        feature_cols = [
            "inflight_wifi_service",
            "seat_comfort",
            "food_and_drink",
            "arrival_delay_in_minutes"
        ]

        spark_df = df.select(*feature_cols, "label")

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )

        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label"
        )

        pipeline = Pipeline(stages=[assembler, lr])

        model = pipeline.fit(spark_df)

        user_df = spark.createDataFrame(
            [(wifi, seat, food, delay)],
            feature_cols
        )

        result = model.transform(user_df).select("prediction", "probability").collect()[0]

        prediction = int(result["prediction"])
        probability = float(result["probability"][1])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Passenger is LIKELY SATISFIED")
        else:
            st.error("Passenger is LIKELY NOT SATISFIED")

        st.write(f"Satisfaction Probability: **{probability*100:.2f}%**")

        st.subheader("Service Recommendations")

        if wifi <= 2:
            st.write("➡ Improve inflight WiFi service")
        if seat <= 2:
            st.write("➡ Enhance seat comfort")
        if food <= 2:
            st.write("➡ Improve food quality")
        if delay > 60:
            st.write("➡ Reduce flight delays")