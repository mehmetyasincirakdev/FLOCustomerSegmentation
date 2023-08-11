###############################################################

# Customer Segmentation with RFM
###############################################################
# Business Problem
###############################################################
# FLO wants to segment its customers and define marketing strategies based on these segments.
# In order to achieve this, customer behaviors will be defined and groups will be created based on these behavioral clusters.
###############################################################
# Data Set Story
###############################################################
# The dataset consists of information derived from the past shopping behaviors of customers who made their most recent purchases in the years 2020 - 2021 through the OmniChannel (both online and offline shopping).
# master_id: Unique customer number
# order_channel: The channel used for shopping (Android, ios, Desktop, Mobile, Offline)
# last_order_channel: The channel used for the most recent purchase
# first_order_date: The date of the customer's first purchase
# last_order_date: The date of the customer's most recent purchase
# last_order_date_online: The date of the customer's most recent online purchase
# last_order_date_offline: The date of the customer's most recent offline purchase
# order_num_total_ever_online: Total number of purchases made by the customer online
# order_num_total_ever_offline: Total number of purchases made by the customer offline
# customer_value_total_ever_offline: Total amount spent by the customer in offline purchases
# customer_value_total_ever_online: Total amount spent by the customer in online purchases
# interested_in_categories_12: List of categories the customer has shopped from in the last 12 months
################################################################
###############################################################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Understanding)
###############################################################
# TASKS
###############################################################

# TASK 1: Data Understanding and Preparation
# 1. Read the flo_data_20K.csv dataset.
# 2. In the dataset,
#   a. Display the first 10 observations,
#   b. List variable names,
#   c. Provide descriptive statistics,
#   d. Identify missing values,
#   e. Examine variable types.
# 3. Create new variables for the total number of purchases and spending for each customer who is an Omnichannel shopper (both online and offline).
# 4. Examine variable types. Convert date-related variables to the date data type.
# 5. Explore the distribution of the number of customers, average number of items bought, and average spending across different shopping channels.
# 6. List the top 10 customers with the highest spending.
# 7. List the top 10 customers with the most orders.
# 8. Functionally prepare the data preprocessing steps.
# TASK 2: Calculation of RFM Metrics
# TASK 3: Calculation of RF and RFM Scores
# TASK 4: Defining RF Scores as Segments
# TASK 5: Action Time!
# 1. Analyze the recency, frequency, and monetary averages of the segments.
# 2. Use RFM analysis to find customers who fit the profiles for 2 cases and save their customer IDs to a CSV file.
# a. FLO is introducing a new women's shoe brand. The prices of this brand's products are above the general customer preferences. Therefore, customers who are loyal (champions, loyal_customers), spend an average of more than 250 TL, and purchase from the women's category will be contacted for the promotion and sales of this brand. Save the customer IDs to a CSV file named new_brand_target_customers.csv.
# b. A discount of nearly 40% is planned for men's and children's products. Customers who were good customers in the past but haven't shopped for a long time, "sleeping" customers, and new customers interested in these categories should be targeted. Save the customer IDs of suitable profiles to a CSV file named discount_target_customer_ids.csv.
# TASK 6: Functionize the Entire Process.

import pandas as pd
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

# 1. Read the flo_data_20K.csv dataset. Create a copy of the DataFrame.
df_ = pd.read_csv("Datasets/flo_data_20k.csv")
df = df_.copy()
df.head()

df.head(10)
df.columns
df.shape
df.describe().T
df.isnull().sum()
df.info()

# 3. "Omnichannel" refers to customers who have made purchases from both online and offline platforms.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 4. Examine the variable types. Convert the variables representing dates to the "date" data type.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

# 5. Examine the distribution of the number of customers across shopping channels, total items purchased, and total expenditures.
df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total": "sum",
                                 "customer_value_total": "sum"})

# 6. list the top 10 customers that generate the most revenue.
df.sort_values("customer_value_total", ascending=False)[:10]

# 7. list the top 10 customers that generate the least revenue.
df.sort_values("order_num_total", ascending=False)[:10]


# 8. Functionalize the data preprocessing process.
def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df


###############################################################
# TASK 2: Calculating RFM Metrics
###############################################################

# The analysis date is 2 days after the last purchase in the dataset.
df["last_order_date"].max()  # 2021-05-30
analysis_date = dt.datetime(2021, 6, 1)

# A new rfm dataframe with customer_id, recency, frequnecy and monetary values.
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

rfm.head()

###############################################################
# Task 3: Calculating RF and RFM Scores
###############################################################

#  The Recency, Frequency and Monetary metrics are converted to scores between 1-5 using qcut.
# These scores are saved as recency_score, frequency_score and monetary_score.
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()

# The recency_score and frequency_score are combined into a single variable and saved as RF_SCORE.
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

# 3. The recency_score, frequency_score and monetary_score are combined into a single variable and saved as RFM_SCORE.
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

rfm.head()

###############################################################
# Task 4: Defining RF Scores as Segments
###############################################################

# Define segments to make the created RF scores more explainable, and convert RF_SCORE to segments using the defined seg_map.
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head()

###############################################################
# GÖREV 5: Action time!
###############################################################

# 1. Examine the recency, frequnecy and monetary averages of the segments.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# 2. Find the relevant profile customers for 2 cases with the help of RFM analysis and save the customer IDs to csv.

# a. FLO is adding a new women's shoe brand to its portfolio. The product prices of the brand it includes are above the general customer preferences.
# Therefore, it is desired to communicate with the customers in the profile who will be interested in the promotion and product sales of the brand.
# It is planned that these customers will be loyal and women who shop from the women's category.
# Save the customer ID numbers to the csv file as yeni_marka_hedef_müşteri_id.cvs.

target_segments_customer_ids = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape

rfm.head()

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
#  alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
#  olarak kaydediniz.
target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & (
        (df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)


###############################################################
# BONUS
###############################################################

def create_rfm(dataframe):

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)


    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    rfm = pd.DataFrame()
    rfm["customer_id"] = dataframe["master_id"]
    rfm["recency"] = (analysis_date - dataframe["last_order_date"]).astype('timedelta64[D]')
    rfm["frequency"] = dataframe["order_num_total"]
    rfm["monetary"] = dataframe["customer_value_total"]


    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))


    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

    return rfm[["customer_id", "recency", "frequency", "monetary", "RF_SCORE", "RFM_SCORE", "segment"]]


rfm_df = create_rfm(df)
