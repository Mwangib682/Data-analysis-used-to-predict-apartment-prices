# Import libraries here
import warnings
from glob import glob
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wqet_grader
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

# Build your `wrangle` function
def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)
    # Subset data: Apartments in "Capital Federal", less than 100,000
    mask_ba = df["place_with_parent_names"].str.contains("Distrito Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 100_000
    df = df[mask_ba & mask_apt & mask_price]
    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]
    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)
    # Get place name
    df["borough"] = df["place_with_parent_names"].str.split("|", expand=True)[1]
    df.drop(columns="place_with_parent_names", inplace=True)
    #drop features with high null counts
    df.drop(columns=["floor", "expenses"], inplace = True)
    #drop low- and high- cordianillity categorical variables
    df.drop(columns=["operation", "property_type", "currency", "properati_url"], inplace = True)
    #drop leaky columns
    df.drop(columns = [
        "price",
        "price_aprox_local_currency",
        "price_per_m2",
        "price_usd_per_m2"
    ],
    inplace = True
    )
    #drop columns with multicollinearity
    df.drop(columns = ["surface_total_in_m2", "rooms"], inplace = True)
    #dropna values
    #df.dropna(inplace=True)
    return df

#Use glob to create the list files
files = glob("data/mexico-city-real-estate-*.csv")
files

#Combine your wrangle function, a list comprehension, and pd.concat to create a DataFrame df
frames = [wrangle(file) for file in files]
df = pd.concat(frames, ignore_index = True)
print(df.info())
df.head()

#Create a histogram showing the distribution of apartment prices ("price_aprox_usd") in df
plt.hist(df["price_aprox_usd"])
plt.xlabel("Price [$]")
plt.ylabel("Count")
plt.title("Distribution of Apartment Prices");

#Create a scatter plot that shows apartment price ("price_aprox_usd") as a function of apartment size ("surface_covered_in_m2")
plt.scatter(x = df["surface_covered_in_m2"], y = df ["price_aprox_usd"])
plt.xlabel("Area[sq meters]")
plt.ylabel("price[USD]")
plt.title("Mexico City: Price vs Area");

#Create a Mapbox scatter plot that shows the location of the apartments in your dataset and represent their price using color.
fig = px.scatter_mapbox(
    df,  # Our DataFrame
    lat="lat",
    lon="lon",
    width=600,  # Width of map
    height=600,  # Height of map
    color= "price_aprox_usd",
    hover_data=["price_aprox_usd"],  # Display price when hovering mouse over house
)
fig.update_layout(mapbox_style="open-street-map")
fig.show()

#Calculate the baseline mean absolute error for your model.
y_mean = y_train.mean()
y_pred_baseline = [y_mean]*len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
print("Mean apt price:", round(y_mean,2))
print("Baseline MAE:", mean_absolute_error(y_train, y_pred_baseline))

#Create a pipeline named model
# Build Model
model = make_pipeline(
    OneHotEncoder(use_cat_names = True), 
    SimpleImputer(),
    Ridge()
)
model.fit(X_train, y_train)

#Read the CSV file mexico-city-test-features.csv into the DataFrame X_test
X_test = pd.read_csv("data/mexico-city-test-features.csv")
print(X_test.info())
X_test.head()

#Use your model to generate a Series of predictions for X_test
y_test_pred = pd.Series(model.predict(X_test))
y_test_pred.head()

#Create a Series named feat_imp
intercept = model.named_steps["ridge"].intercept_
coefficients = model.named_steps["ridge"].coef_
feature_names = model.named_steps["onehotencoder"].get_feature_names()
feat_imp = pd.Series(coefficients, index = feature_names)
feat_imp.head()

# Build bar chart
feat_imp.sort_values(key=abs).tail(10).plot(kind = "barh")
plt.xlabel("Importance[USD]")
plt.ylabel("Feature")
plt.title("Feature Importance for Apartment Price");

