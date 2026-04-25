import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import matplotlib.ticker as mtick

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    scorecard   = pd.read_csv('customer_scorecard.csv')
    shap_vals   = pd.read_csv('shap_values.csv')
    X_test_shap = pd.read_csv('X_test_with_shap.csv')
    return scorecard, shap_vals, X_test_shap

@st.cache_resource
def load_model():
    with open('churn_model.pkl', 'rb') as f:
        return pickle.load(f)

scorecard, shap_vals, X_test_shap = load_data()
model = load_model()

st.sidebar.title("📊 Churn Prediction")
st.sidebar.markdown("IBM Telco Customer Dataset")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Customer Risk Scorecard",
     "Individual Customer Analysis", "Retention ROI Calculator"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** XGBoost")
st.sidebar.markdown(f"**Customers analyzed:** {len(scorecard):,}")
high_risk = scorecard[scorecard['risk_tier'] == 'HIGH']
st.sidebar.markdown(f"**High risk customers:** {len(high_risk):,}")
st.sidebar.markdown(
    f"**Revenue at risk:** ${high_risk['CLV'].sum():,.0f}"
)

# PAGE 1 — OVERVIEW
if page == "Overview":
    st.title("Customer Churn Prediction Dashboard")
    st.markdown("*Predict · Explain · Prioritize · Retain*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(scorecard):,}")
    with col2:
        high = len(scorecard[scorecard['risk_tier']=='HIGH'])
        st.metric("High Risk", f"{high:,}",
                  delta=f"{high/len(scorecard)*100:.1f}% of total",
                  delta_color="inverse")
    with col3:
        revenue = scorecard[scorecard['risk_tier']=='HIGH']['CLV'].sum()
        st.metric("Revenue at Risk", f"${revenue:,.0f}")
    with col4:
        avg_prob = scorecard['churn_probability'].mean()
        st.metric("Avg Churn Probability", f"{avg_prob*100:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Tier Distribution")
        tier_counts = scorecard['risk_tier'].value_counts()
        colors = {'HIGH':'#F44336','MEDIUM':'#FF9800','LOW':'#4CAF50'}
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(tier_counts.index, tier_counts.values,
               color=[colors[t] for t in tier_counts.index],
               width=0.5, edgecolor='white')
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x()+p.get_width()/2, p.get_height()),
                        ha='center', va='bottom', fontsize=11)
        ax.set_ylabel('Number of Customers')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Churn Probability Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(scorecard['churn_probability'], bins=30,
                color='#2196F3', edgecolor='white', alpha=0.8)
        ax.axvline(x=0.5, color='#F44336', linestyle='--',
                   linewidth=1.5, label='Decision threshold (0.5)')
        ax.axvline(x=0.7, color='#FF9800', linestyle='--',
                   linewidth=1.5, label='High risk threshold (0.7)')
        ax.set_xlabel('Churn Probability')
        ax.set_ylabel('Number of Customers')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Key Insights from SHAP Analysis")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.info("**#1 Driver: Contract Type**\n\nMonth-to-month "
                "customers churn at 3× the rate of annual contract customers.")
    with i2:
        st.warning("**#2 Driver: Tenure**\n\nCustomers in their "
                   "first 12 months are the highest risk group.")
    with i3:
        st.error("**#3 Driver: Charge per Service**\n\nCustomers "
                 "paying high monthly charges for few services feel "
                 "poor value.")

# PAGE 2 — CUSTOMER RISK SCORECARD
elif page == "Customer Risk Scorecard":
    st.title("Customer Risk Scorecard")
    st.markdown("All customers ranked by priority score (Churn Probability × CLV)")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        tier_filter = st.multiselect(
            "Filter by Risk Tier",
            ['HIGH','MEDIUM','LOW'],
            default=['HIGH','MEDIUM']
        )
    with col2:
        contract_filter = st.multiselect(
            "Filter by Contract",
            scorecard['Contract'].unique().tolist(),
            default=scorecard['Contract'].unique().tolist()
        )
    with col3:
        top_n = st.slider("Show top N customers", 10, 200, 50)

    filtered = scorecard[
        (scorecard['risk_tier'].isin(tier_filter)) &
        (scorecard['Contract'].isin(contract_filter))
    ].head(top_n)

    st.markdown(f"Showing **{len(filtered)}** customers")

    def color_tier(val):
        colors = {'HIGH':'background-color:#FFEBEE;color:#C62828',
                  'MEDIUM':'background-color:#FFF3E0;color:#E65100',
                  'LOW':'background-color:#E8F5E9;color:#1B5E20'}
        return colors.get(val, '')

    display_cols = ['priority_rank', 'risk_tier', 'churn_probability',
                    'CLV', 'priority_score', 'Contract',
                    'tenure', 'MonthlyCharges', 'top_risk_factors']

    # ✅ FIXED — applymap changed to map
    styled = filtered[display_cols].style\
        .map(color_tier, subset=['risk_tier'])\
        .format({'churn_probability': '{:.1%}',
                 'CLV': '${:,.0f}',
                 'priority_score': '{:,.0f}',
                 'MonthlyCharges': '${:.2f}'})

    st.dataframe(styled, use_container_width=True, height=500)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total CLV at Risk",
                  f"${filtered['CLV'].sum():,.0f}")
    with c2:
        st.metric("Avg Churn Probability",
                  f"{filtered['churn_probability'].mean()*100:.1f}%")
    with c3:
        st.metric("Avg Monthly Charges",
                  f"${filtered['MonthlyCharges'].mean():,.2f}")

# PAGE 3 — INDIVIDUAL CUSTOMER ANALYSIS
elif page == "Individual Customer Analysis":
    st.title("Individual Customer Analysis")
    st.markdown("Deep dive into any customer's churn risk with SHAP explanation")
    st.markdown("---")

    customer_idx = st.slider(
        "Select Customer (by priority rank)",
        1, len(scorecard), 1
    )

    customer = scorecard.iloc[customer_idx - 1]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Customer Profile")
        prob  = customer['churn_probability']
        tier  = customer['risk_tier']
        color = {'HIGH':'🔴','MEDIUM':'🟡','LOW':'🟢'}[tier]

        st.markdown(f"**Risk Level:** {color} {tier}")
        st.markdown(f"**Churn Probability:** {prob*100:.1f}%")
        st.markdown(f"**CLV:** ${customer['CLV']:,.0f}")
        st.markdown(f"**Priority Score:** {customer['priority_score']:,.0f}")
        st.markdown("---")
        st.markdown(f"**Contract:** {customer['Contract']}")
        st.markdown(f"**Tenure:** {customer['tenure']:.0f} months")
        st.markdown(f"**Monthly Charges:** ${customer['MonthlyCharges']:.2f}")
        st.markdown(f"**Internet Service:** {customer['InternetService']}")
        st.markdown("---")
        st.markdown("**Top Risk Factors:**")
        for factor in customer['top_risk_factors'].split(', ')[:3]:
            st.markdown(f"  ⚠️ {factor}")

    with col2:
        st.subheader("SHAP Explanation — Why is this customer at risk?")

        shap_row = shap_vals.drop(
            columns=['churn_probability','top_risk_factors'],
            errors='ignore'
        ).iloc[customer_idx - 1]

        top_shap = shap_row.abs().nlargest(10)
        shap_vals_plot = shap_row[top_shap.index]

        colors_shap = ['#F44336' if v > 0 else '#2196F3'
                       for v in shap_vals_plot.values]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(shap_vals_plot)),
                shap_vals_plot.values,
                color=colors_shap, alpha=0.85,
                edgecolor='white')
        ax.set_yticks(range(len(shap_vals_plot)))
        ax.set_yticklabels(shap_vals_plot.index, fontsize=10)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Value (impact on churn prediction)')
        ax.set_title('Red = pushes toward churn | Blue = pushes away',
                     fontsize=10, style='italic')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Recommended Action")
    if tier == 'HIGH':
        st.error(
            f"🚨 **Immediate action required.** This customer has a "
            f"{prob*100:.1f}% churn probability and represents "
            f"${customer['CLV']:,.0f} in CLV.\n\n"
            f"**Suggested intervention:** Personal outreach, contract "
            f"upgrade offer, or loyalty discount within 48 hours."
        )
    elif tier == 'MEDIUM':
        st.warning(
            f"⚠️ **Monitor closely.** Churn probability is {prob*100:.1f}%."
            f"\n\n**Suggested intervention:** Send targeted email offer "
            f"or service upgrade suggestion."
        )
    else:
        st.success(
            f"✅ **Low risk.** Churn probability is only {prob*100:.1f}%."
            f"\n\nNo immediate action needed — focus resources elsewhere."
        )

# PAGE 4 — RETENTION ROI CALCULATOR
elif page == "Retention ROI Calculator":
    st.title("Retention ROI Calculator")
    st.markdown("How much revenue can we save by acting on at-risk customers?")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Adjust Assumptions")
        n_customers = st.slider(
            "Number of customers to contact", 10, 300, 100
        )
        retention_rate = st.slider(
            "Expected retention rate (%)", 10, 60, 30
        ) / 100
        incentive_cost = st.slider(
            "Incentive cost per customer ($)", 10, 200, 50
        )

    top_n       = scorecard.head(n_customers)
    rev_at_risk = top_n['CLV'].sum()
    rev_saved   = rev_at_risk * retention_rate
    total_cost  = n_customers * incentive_cost
    net_roi     = rev_saved - total_cost
    roi_pct     = (net_roi / total_cost) * 100

    with col2:
        st.subheader("Results")
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Revenue at Risk",  f"${rev_at_risk:,.0f}")
            st.metric("Revenue Saved",    f"${rev_saved:,.0f}")
        with r2:
            st.metric("Intervention Cost", f"${total_cost:,.0f}")
            st.metric("Net ROI",
                      f"${net_roi:,.0f}",
                      delta=f"{roi_pct:.0f}% return",
                      delta_color="normal")

    st.markdown("---")
    st.subheader("ROI Across Different Intervention Sizes")

    ns       = list(range(10, 310, 10))
    net_rois = []
    for n in ns:
        tn    = scorecard.head(n)
        saved = tn['CLV'].sum() * retention_rate
        cost  = n * incentive_cost
        net_rois.append(saved - cost)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ns, net_rois, color='#2196F3', linewidth=2.5)
    ax.fill_between(ns, net_rois, alpha=0.15, color='#2196F3')
    ax.axhline(y=0, color='#F44336', linestyle='--',
               linewidth=1.2, label='Break-even')
    ax.axvline(x=n_customers, color='#FF9800', linestyle='--',
               linewidth=1.2, label=f'Your selection ({n_customers})')
    ax.set_xlabel('Number of Customers Contacted')
    ax.set_ylabel('Net ROI ($)')
    ax.set_title('Net ROI by Intervention Size', fontsize=12)
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, p: f'${x:,.0f}')
    )
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.caption(
        "Assumptions: CLV = MonthlyCharges × expected remaining tenure. "
        "Retention rate and incentive cost are adjustable estimates. "
        "Actual results may vary."
    )