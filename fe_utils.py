import pandas as pd
import numpy as np

def abc_categorizer(df):
    """
    Phân loại ABC cho danh mục sản phẩm dựa trên doanh thu.
    """
    grouped_df = (
        df.groupby("Product_Category", as_index=False)["Total_Revenue"]
          .sum()
          .rename(columns={"total_revenue": "cat_revenue"})
    )
    grouped_df = grouped_df.sort_values("cat_revenue", ascending=False)

    total_rev = grouped_df["cat_revenue"].sum()
    grouped_df["share"]     = grouped_df["cat_revenue"] / total_rev
    grouped_df["cum_share"] = grouped_df["share"].cumsum()

    def assign_class(cum):
        if cum <= 0.80:
            return "A"
        elif cum <= 0.95:
            return "B"
        else:
            return "C"

    grouped_df["ABC"] = grouped_df["cum_share"].apply(assign_class)
    print(grouped_df[["Product_Category", "cat_revenue", "cum_share", "ABC"]].to_string(index=False))
    return grouped_df.set_index("Product_Category")["ABC"].to_dict()


def heuristic_segment(score):
    """
    Phân khúc khách hàng dựa trên điểm RFM.
    """
    if score <= 5:
        return "Standard"    # 3-5
    elif score <= 8:
        return "Silver"      # 6-8
    elif score <= 10:
        return "Premium"     # 9-10
    else:
        return "Gold"        # 11-12

def heuristic_lrfm_segment(score):
    """
    Phân khúc khách hàng dựa trên điểm LRFM (tối đa 16 điểm).
    """
    if score <= 7:
        return "Standard"    # 4-7
    elif score <= 11:
        return "Silver"      # 8-11
    elif score <= 14:
        return "Premium"     # 12-14
    else:
        return "Gold"        # 15-16


def apply_basic_features(df):
    """
    Áp dụng các bước Feature Engineering từ 3.1 đến 3.4.
    Bao gồm: Time Features, Coupon Flags, Invoice, Total Revenue.
    Lưu ý: df đầu vào cần có đủ các cột tương ứng (từ việc merge trước đó).
    """
    df = df.copy()
    
    # 3.1 Time Features
    if "Transaction_Date" in df.columns:
        df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
        df["Month"]            = df["Transaction_Date"].dt.strftime("%b")
        df["Date"]             = df["Transaction_Date"].dt.date
        df["Week"]             = df["Transaction_Date"].dt.to_period("W").astype(str)
        
    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].astype(str)
    if "Transaction_ID" in df.columns:
        df["Transaction_ID"] = df["Transaction_ID"].astype(str)
    
    # 3.2 Coupon Flags
    if "Discount_pct" in df.columns:
        df["Is_Discounted"]  = (df["Discount_pct"] > 0).astype(int)
    if "Coupon_Status" in df.columns:
        df["Is_Used_Coupon"] = (df["Coupon_Status"] == "Used").astype(int)
    
    if df['Discount_pct'].max() > 1:
        df['Discount_pct'] = df['Discount_pct'] / 100
    
    # 3.3 Invoice
    cols_invoice = {"Quantity", "Avg_Price", "Discount_pct", "GST", "Delivery_Charges", "Coupon_Status"}
    if cols_invoice.issubset(df.columns):
        df["Invoice"] = np.where(
            df["Coupon_Status"] == "Used",
            (df["Quantity"] * df["Avg_Price"]) * (1 - df["Discount_pct"]) * (1 + df["GST"]) + df["Delivery_Charges"],
            (df["Quantity"] * df["Avg_Price"]) * (1 + df["GST"]) + df["Delivery_Charges"]
        )
        
    # 3.4 Total Revenue
    cols_revenue = {"Quantity", "Avg_Price", "Discount_pct", "GST"}
    if cols_revenue.issubset(df.columns):
        df["Total_Revenue"] = (
            df["Quantity"] * df["Avg_Price"]
            * (1 - df["Discount_pct"])
            * (1 + df["GST"])
        )
        
    return df


def calculate_lrfm(df):
    """
    Mục 3.6: Tính toán các chỉ số LRFM và gộp vào DataFrame gốc.
    Yêu cầu: df phải có các cột 'CustomerID', 'Transaction_Date', 'Transaction_ID', 'Invoice'.
    Length (L): Số ngày từ lần mua đầu tiên đến hiện tại.
    """
    df = df.copy()
    
    if not {"CustomerID", "Transaction_Date", "Transaction_ID", "Invoice"}.issubset(df.columns):
        print("Warning: Không đủ các cột bắt buộc (CustomerID, Transaction_Date, Transaction_ID, Invoice) để tính LRFM.")
        return df, None

    today = df["Transaction_Date"].max() + pd.Timedelta(days=1)

    lrfm = df.groupby("CustomerID").agg(
        Length    = ("Transaction_Date", lambda x: (today - x.min()).days),
        Recency   = ("Transaction_Date", lambda x: (today - x.max()).days),
        Frequency = ("Transaction_ID", "nunique"),
        Monetary  = ("Invoice", "sum")
    ).reset_index()

    # Tính LRFM scores (dao chiều Recency)
    lrfm["L_Score"] = pd.qcut(lrfm["Length"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    lrfm["R_Score"] = pd.qcut(lrfm["Recency"],  q=4, labels=[4, 3, 2, 1]).astype(int)
    lrfm["F_Score"] = pd.qcut(lrfm["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    lrfm["M_Score"] = pd.qcut(lrfm["Monetary"].rank(method="first"),  q=4, labels=[1, 2, 3, 4]).astype(int)

    lrfm["LRFM_Score"] = lrfm["L_Score"] + lrfm["R_Score"] + lrfm["F_Score"] + lrfm["M_Score"]
    lrfm["Heuristic_Segment"] = lrfm["LRFM_Score"].apply(heuristic_lrfm_segment)

    # Trước khi merge, xoá các cột này nếu đã tồn tại để tránh lỗi khi chạy lại (MergeError)
    cols_lrfm = [
        "Length", "Recency", "Frequency", "Monetary",
        "L_Score", "R_Score", "F_Score", "M_Score", "LRFM_Score", "Heuristic_Segment"
    ]
    existing_cols = [c for c in cols_lrfm if c in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)

    # Gán LRFM & segment vào df
    df = df.merge(
        lrfm[["CustomerID"] + cols_lrfm],
        on="CustomerID",
        how="left"
    )
    return df, lrfm



def aggregate_marketing_monthly(df):
    """
    Mục 3.7: Tạo bảng tổng hợp chi phí marketing theo tháng.
    """
    required_cols = {"Month", "Invoice", "Discount_pct", "GST", "Delivery_Charges", "Total_Marketing_Spend"}
    if not required_cols.issubset(df.columns):
        print("Warning: df thiếu một số cột để tổng hợp marketing monthly.")
        return None

    marketing_monthly = (
        df.groupby("Month", as_index=False)
          .agg(
              Invoice         = ("Invoice", "sum"),
              Discount_pct    = ("Discount_pct", "mean"),
              GST             = ("GST", "mean"),
              DeliveryCharges = ("Delivery_Charges", "sum"),
              Total_Mkt_Spend = ("Total_Marketing_Spend", "sum")
          )
    )

    marketing_monthly["Pct_Mkt_Spend"] = (
        marketing_monthly["Total_Mkt_Spend"] * 100 / marketing_monthly["Invoice"]
    )
    marketing_monthly["Pct_Delivery"] = (
        marketing_monthly["DeliveryCharges"] * 100 / marketing_monthly["Invoice"]
    )
    
    return marketing_monthly

def calculate_clv(df):
    """
    Tính toán các chỉ số AOV, Purchase Frequency, Churn Rate và CLV.
    """
    df_result = df.copy()
    
    # 1. Tính AOV và Purchase Frequency cho từng khách hàng
    # Giả định: 
    # - TotalPrice = Tổng Invoice (hoặc Total_Revenue)
    # - Total Transactions = Số lượng Transaction_ID duy nhất
    # - Periods of Time = Max của Tenure_Months (hoặc cố định 12 tháng)
    
    customer_agg = df_result.groupby('CustomerID').agg(
        Total_Price=('Invoice', 'sum'),
        Total_Transactions=('Transaction_ID', 'nunique'),
        Periods_of_Time=('Tenure_Months', 'max')
    ).reset_index()
    
    # Xử lý trường hợp Periods_of_Time = 0 để tránh lỗi chia 0
    customer_agg['Periods_of_Time'] = customer_agg['Periods_of_Time'].replace(0, 1)
    
    customer_agg['AOV'] = customer_agg['Total_Price'] / customer_agg['Total_Transactions']
    customer_agg['Purchase_Frequency'] = customer_agg['Total_Transactions'] / customer_agg['Periods_of_Time']
    
    # 2. Tính Churn Rate tổng thể (Global Churn Rate)
    # Churn Rate = (customers leaving in period / customers buyed in the period before)
    # Ở đây ta lấy trung bình theo từng tháng
    if 'Transaction_Date' in df_result.columns:
        df_result['Month_Year'] = df_result['Transaction_Date'].dt.to_period('M')
        monthly_customers = df_result.groupby('Month_Year')['CustomerID'].unique()
        
        churn_rates = []
        months = sorted(monthly_customers.index)
        for i in range(1, len(months)):
            prev_month_cust = set(monthly_customers[months[i-1]])
            curr_month_cust = set(monthly_customers[months[i]])
            
            buyed_in_prev = len(prev_month_cust)
            leaving_in_curr = len(prev_month_cust - curr_month_cust)
            
            if buyed_in_prev > 0:
                churn_rates.append(leaving_in_curr / buyed_in_prev)
                
        # Tính churn rate trung bình
        global_churn_rate = np.mean(churn_rates) if churn_rates else 0.1
        
        # Chia cho 100 theo công thức của bạn (lưu ý: tuỳ thuộc vào việc bạn coi tỷ lệ trên đã là % hay chưa)
        global_churn_rate = global_churn_rate / 100
        
        # Tránh chia cho 0
        if global_churn_rate == 0:
            global_churn_rate = 0.0001
    else:
        global_churn_rate = 0.01 # Mặc định nếu không tính được
        
    customer_agg['Churn_Rate'] = global_churn_rate
    
    # 3. Tính CLV
    customer_agg['CLV'] = customer_agg['AOV'] * customer_agg['Purchase_Frequency'] * (1 / customer_agg['Churn_Rate'])
    
    # 4. Gộp các cột này vào df gốc
    cols_to_merge = ['AOV', 'Purchase_Frequency', 'Churn_Rate', 'CLV']
    
    # Xoá các cột này nếu đã tồn tại trong df_result để tránh lỗi MergeError khi chạy nhiều lần
    existing_cols = [c for c in cols_to_merge if c in df_result.columns]
    if existing_cols:
        df_result = df_result.drop(columns=existing_cols)
        
    df_result = df_result.merge(customer_agg[['CustomerID'] + cols_to_merge], on='CustomerID', how='left')

    
    # Xoá cột tạm
    if 'Month_Year' in df_result.columns:
        df_result = df_result.drop(columns=['Month_Year'])
        
    return df_result

